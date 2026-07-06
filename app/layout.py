"""Detects the vertical bands of music systems on a (preprocessed) sheet
image, so the client can highlight the line being played in the score
view. Pure image processing — no Audiveris internals: staff lines are
near-full-width dark rows, which makes a horizontal projection profile a
very reliable detector on the deskewed images this pipeline produces."""

import io
import logging

import numpy as np
from PIL import Image

log = logging.getLogger("snapnotes.layout")

# Work on a bounded-size thumbnail; bands are returned normalized so the
# original resolution is irrelevant.
_ANALYSIS_WIDTH = 1000

# A row counts as a "staff-line row" when at least this fraction of its
# pixels are ink. Staff lines span most of the page width; note stems,
# beams, and text never get close on a full-width row basis.
_ROW_INK_THRESHOLD = 0.35


def detect_system_bands(
    png_bytes: bytes, staves_per_system: int
) -> list[dict]:
    """Return one dict per music system, top to bottom — or [] when
    detection isn't confident (caller falls back to page-level sync):

        {"y0", "y1": normalized vertical band,
         "x0", "x1": normalized horizontal extent of the staff lines}

    The x extent drives the client's moving playhead: beats map onto the
    line between x0 and x1. `staves_per_system` comes from the MusicXML
    part count (a piano grand staff arrives from Audiveris as two
    parts)."""
    try:
        im = Image.open(io.BytesIO(png_bytes))
        im.load()
    except Exception:  # noqa: BLE001
        return []
    if im.mode != "L":
        im = im.convert("L")
    w, h = im.size
    if w > _ANALYSIS_WIDTH:
        im = im.resize(
            (_ANALYSIS_WIDTH, max(1, round(h * _ANALYSIS_WIDTH / w))),
            Image.BILINEAR,
        )
    arr = np.asarray(im, dtype=np.uint8)
    height = arr.shape[0]
    if height < 40:
        return []

    # Ink fraction per row, measured over the content's horizontal extent
    # (staff lines span the *music*, not necessarily the page margins).
    full_ink = arr < 200
    col_has_ink = full_ink.mean(axis=0) > 0.01
    cols = np.where(col_has_ink)[0]
    if len(cols) < 20:
        return []
    width = arr.shape[1]
    ink = full_ink[:, cols[0] : cols[-1] + 1]
    row_ratio = ink.mean(axis=1)

    line_rows = np.where(row_ratio >= _ROW_INK_THRESHOLD)[0]
    if len(line_rows) < 5 * max(1, staves_per_system):
        return []

    # Runs of adjacent rows = individual staff lines (a line is 1–4 rows
    # thick at this scale). Collect each line's center.
    centers: list[float] = []
    run_start = line_rows[0]
    prev = line_rows[0]
    for r in line_rows[1:]:
        if r > prev + 2:
            centers.append((run_start + prev) / 2.0)
            run_start = r
        prev = r
    centers.append((run_start + prev) / 2.0)
    if len(centers) < 5:
        return []

    # Interline = the typical gap between adjacent staff lines. Use the
    # median of the smaller half of gaps (gaps between staves/systems
    # pollute the upper half).
    gaps = np.diff(centers)
    small_gaps = np.sort(gaps)[: max(1, len(gaps) // 2)]
    interline = float(np.median(small_gaps))
    if interline <= 0:
        return []

    # Group lines into staves: consecutive lines whose gap is staff-like.
    staves: list[tuple[float, float]] = []  # (top line, bottom line)
    start = centers[0]
    prev_c = centers[0]
    count = 1
    for c in centers[1:]:
        if c - prev_c <= interline * 1.8:
            prev_c = c
            count += 1
        else:
            if count >= 4:  # tolerate one merged/missed line per staff
                staves.append((start, prev_c))
            start = c
            prev_c = c
            count = 1
    if count >= 4:
        staves.append((start, prev_c))

    if not staves:
        return []

    # Group staves into systems. Preferred: the MusicXML told us how many
    # staves each system has and the count divides cleanly. Fallback: gap
    # clustering (intra-system staff gaps are smaller than inter-system
    # gaps).
    n = len(staves)
    sps = max(1, staves_per_system)
    if n % sps == 0:
        grouped = [staves[i : i + sps] for i in range(0, n, sps)]
    elif sps > 1:
        grouped = _group_by_gaps(staves)
        if not grouped:
            return []
    else:
        grouped = [[s] for s in staves]

    pad = interline * 2.0
    bands: list[list[float]] = []
    extents: list[tuple[float, float]] = []
    barlines: list[list[float]] = []
    for group in grouped:
        y0 = max(0.0, group[0][0] - pad)
        y1 = min(float(height - 1), group[-1][1] + pad)
        bands.append([y0 / height, y1 / height])
        # Horizontal extent of this system's staff lines: ink columns
        # within the (unpadded) staff rows. Drives the playhead range.
        r0 = int(group[0][0])
        r1 = int(group[-1][1]) + 1
        band_cols = np.where(full_ink[r0:r1].mean(axis=0) > 0.02)[0]
        if len(band_cols) >= 10:
            extents.append(
                (band_cols[0] / width, (band_cols[-1] + 1) / width)
            )
        else:
            extents.append((cols[0] / width, (cols[-1] + 1) / width))
        barlines.append(
            _detect_barlines(full_ink, r0, r1, interline, width)
        )

    # Sanity: bands must be ordered and non-overlapping (small overlaps
    # from generous padding are clipped to the midpoint).
    for i in range(1, len(bands)):
        if bands[i][0] < bands[i - 1][1]:
            mid = (bands[i][0] + bands[i - 1][1]) / 2.0
            bands[i - 1][1] = mid
            bands[i][0] = mid
    return [
        {
            "y0": round(b[0], 4),
            "y1": round(b[1], 4),
            "x0": round(e[0], 4),
            "x1": round(e[1], 4),
            # Candidate barline x-positions (normalized). main.py keeps
            # them only when the count agrees with the MusicXML measure
            # count; the client's playhead then interpolates between the
            # *actual* barlines instead of assuming even spacing.
            "bars": bars,
        }
        for b, e, bars in zip(bands, extents, barlines)
    ]


def _detect_barlines(
    full_ink: np.ndarray, r0: int, r1: int, interline: float, width: int
) -> list[float]:
    """Barline x-centers for the system occupying rows [r0, r1): thin
    columns whose ink spans nearly the whole system height. Note stems
    only cross one staff (and never the gap between staves of a grand
    staff / choir system), so a high full-height threshold separates
    barlines cleanly. Returns normalized x centers, left to right."""
    if r1 - r0 < 8:
        return []
    col_ratio = full_ink[r0:r1].mean(axis=0)
    candidates = np.where(col_ratio >= 0.82)[0]
    if len(candidates) == 0:
        return []
    max_run = max(3.0, interline * 0.7)  # barlines are thin; blobs aren't
    centers: list[float] = []
    run_start = candidates[0]
    prev = candidates[0]
    for c in candidates[1:]:
        if c > prev + 2:
            if prev - run_start <= max_run:
                centers.append((run_start + prev) / 2.0)
            run_start = c
        prev = c
    if prev - run_start <= max_run:
        centers.append((run_start + prev) / 2.0)

    # Collapse clusters: a line's opening (bracket + brace + barline) and
    # closing (final double bar) produce several full-height verticals a
    # few pixels apart that are one musical boundary. Keep the rightmost
    # of each cluster — that's the barline proper.
    cluster_gap = max(6.0, interline * 1.4)
    collapsed: list[float] = []
    for c in centers:
        if collapsed and c - collapsed[-1] <= cluster_gap:
            collapsed[-1] = c
        else:
            collapsed.append(c)
    return [round(c / width, 4) for c in collapsed]


def _group_by_gaps(
    staves: list[tuple[float, float]],
) -> list[list[tuple[float, float]]]:
    """Cluster staves into systems by the gaps between them: gaps within
    a system (grand-staff spacing) are markedly smaller than gaps between
    systems. Splits at gaps larger than 1.6× the smallest gap. Returns []
    when there's only one gap size to compare (no confidence)."""
    if len(staves) < 2:
        return [staves]
    gaps = [staves[i + 1][0] - staves[i][1] for i in range(len(staves) - 1)]
    smallest = min(gaps)
    if smallest <= 0:
        return []
    groups: list[list[tuple[float, float]]] = [[staves[0]]]
    for stave, gap in zip(staves[1:], gaps):
        if gap > smallest * 1.6:
            groups.append([stave])
        else:
            groups[-1].append(stave)
    return groups
