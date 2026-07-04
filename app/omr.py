"""Runs Audiveris (Optical Music Recognition) on an image and returns the
MusicXML output path. Invoked as a subprocess so a JVM crash, OOM, or hang
doesn't take down the FastAPI server."""

import io
import logging
import os
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

# Pillow raises DecompressionBombError above MAX_IMAGE_PIXELS but only
# *warns* between MAX_IMAGE_PIXELS/2 and MAX_IMAGE_PIXELS. Promote that
# warning to an exception so half-bomb images get rejected on the same
# path as full bombs (the OmrError handler turns either into a friendly
# 422). Applied module-wide; benign images never hit the threshold.
warnings.simplefilter("error", Image.DecompressionBombWarning)

log = logging.getLogger("snapnotes.omr")

# Audiveris can be slow on complex scores (multiple staves, dense
# notation). The CLI starts a JVM each call; budget includes that warmup.
AUDIVERIS_TIMEOUT_SECONDS = 240

# Audiveris needs ~15 px between staff lines. Cheap phone screenshots /
# cropped images often come in well below that; upscale them so OMR has
# a chance. Bicubic is good enough — Audiveris is tolerant of mild blur
# but not of sub-pixel staff lines.
MIN_LONGEST_EDGE = 2400

# Bumped whenever the OMR pipeline changes in a way that would produce
# different output for the same input bytes (preprocessing tweaks,
# Audiveris flag changes, version bumps). Included in the cache key so
# stale entries are invalidated automatically instead of silently served.
# v14: container's Audiveris pin moved 5.4.x-era → 5.6.2 (upstream removed
# AppImage distribution; the .deb line starts at 5.5) — a different engine
# version produces different recognition output for identical bytes.
PIPELINE_VERSION = "v14-audiveris-5.6.2"

# Audiveris's built-in input-quality profile. "Standard" is the default;
# "Synthetic" tightens thresholds for clean digital scores (Finale/Sibelius
# PDF exports), "Poor" loosens them for scans/photos. Callers can override
# per-request when they know the source.
VALID_QUALITIES = ("Synthetic", "Standard", "Poor")

# Laplacian-variance sharpness floor below which we treat a "Synthetic"
# input as actually blurry (most likely a low-DPI scan wrapped in a PDF
# container) and route it through the full photo-cleanup pipeline
# instead of the minimal one. Calibrated against a 3-score corpus:
#   - true vector PDF (rain):     ~4800
#   - high-DPI raster PDF (brahms): ~6000
#   - 75-DPI scan-in-PDF (naruto):  ~2300
# Threshold of 3000 cleanly separates the populations on this corpus —
# but it's a heuristic and could trip on legitimately clean but sparse
# pages. False positive cost is moderate (extra preprocessing passes
# that mostly no-op); false negative cost is high (blurry input gets
# minimal preprocessing). Tilt toward false positives.
_BLUR_THRESHOLD = 3000.0


class OmrError(RuntimeError):
    """OMR failed in a recoverable way (timeout, bad image, no output)."""


def run_audiveris(
    image_bytes: bytes, quality: str = "Standard"
) -> tuple[Path, Path]:
    """Run Audiveris on raw image bytes. Returns `(workdir, xml_path)` —
    caller is responsible for `shutil.rmtree(workdir)` when done; otherwise
    the input image, .omr container, and audiveris log all leak into /tmp.

    `quality` maps to Audiveris's `Profiles.defaultQuality` enum:
      - "Synthetic": clean digital scores (PDF exports from Finale/Sibelius)
      - "Standard":  default — balanced for typical inputs
      - "Poor":      scans/photos with noise, skew, lighting issues
    """
    if quality not in VALID_QUALITIES:
        raise OmrError(
            f"unknown quality {quality!r}; must be one of {VALID_QUALITIES}"
        )

    workdir = Path(tempfile.mkdtemp(prefix="omr_"))
    try:
        img_path = workdir / "input.png"
        # Synthetic inputs go through minimal preprocessing (grayscale,
        # deskew, upscale-if-small) — the full pipeline's autocontrast +
        # unsharp + auto-crop passes were designed for photos and
        # actively *hurt* clean PDF renders. We confirmed empirically
        # that even when a "PDF" is actually a low-DPI raster scan
        # (~75 DPI effective), routing it through the full pipeline
        # makes Audiveris fail entirely (auto-crop cuts into staves,
        # autocontrast amplifies blur into noise). Better to feed
        # Audiveris the unmodified blurry image than to mangle it.
        if quality == "Synthetic":
            img_path.write_bytes(_preprocess_minimal(image_bytes))
        else:
            img_path.write_bytes(_preprocess(image_bytes))

        # Sharpness is captured purely for diagnostic logging — surfaces
        # in the per-page meta so we can see when a "Synthetic" upload
        # was actually a blurry source. No behavior change tied to this
        # value yet; the right downstream fix (super-resolution or a
        # different engine) is a bigger investment.
        sharpness = _estimate_sharpness(image_bytes)
        if sharpness is not None and sharpness < _BLUR_THRESHOLD:
            log.warning(
                "low input sharpness=%.0f (threshold %.0f) — accuracy "
                "may be limited by source image resolution; consider "
                "re-sourcing this PDF from a vector / higher-DPI version",
                sharpness, _BLUR_THRESHOLD,
            )

        # JVM env. Two macOS-specific flags plus a heap bump:
        #   - apple.awt.UIElement=true → no Dock icon / focus-steal even
        #     though the JVM initializes AWT under -batch.
        #   - java.awt.headless=true → belt-and-suspenders; AWT won't try
        #     to talk to the window server at all.
        #   - -Xmx2g → Audiveris default heap (~256m) OOMs on dense piano
        #     scores. 2 GB is comfortable headroom without being greedy.
        env = os.environ.copy()
        existing = env.get("JAVA_TOOL_OPTIONS", "")
        env["JAVA_TOOL_OPTIONS"] = (
            f"{existing} "
            "-Xmx2g "
            # G1 collector handles Audiveris's allocation pattern (lots of
            # short-lived intermediate buffers from image processing,
            # plus a long-tail of SIG graph nodes that live for the whole
            # recognition pass) noticeably better than the default Parallel
            # GC, with smaller pause spikes during the symbol-classification
            # step.
            "-XX:+UseG1GC "
            # Pre-touch all heap pages at JVM start so we don't pay
            # page-fault costs partway through a scan. Cheap one-time
            # cost (~tens of ms for a 2 GB heap) for steadier per-scan
            # timing.
            "-XX:+AlwaysPreTouch "
            "-Dapple.awt.UIElement=true "
            "-Djava.awt.headless=true"
        ).strip()

        # Point Audiveris's Tess4J binding at a Tesseract data directory.
        # Critical detail: Audiveris's TesseractOrder requests LEGACY
        # engine mode (OEM_TESSERACT_ONLY), which requires a `.traineddata`
        # file that contains the classical / legacy classifier — not the
        # LSTM-only data that Homebrew ships by default (~4 MB
        # eng.traineddata). Without a legacy-capable file, OCR silently
        # produces nothing: no tempo text ("Allegro" / "♩=120"), no
        # dynamics text ("mf" / "ff" / "sfz") in the resulting MusicXML.
        #
        # We bundle the combined legacy + LSTM `eng.traineddata` (~22 MB)
        # in `server/tessdata/` and prefer it over system paths so this
        # works the same way on every dev machine and in Docker. Falls
        # back to Homebrew / Linux paths only if our bundled copy is
        # missing (e.g. someone hasn't pulled the LFS file).
        bundled_tessdata = Path(__file__).resolve().parents[1] / "tessdata"
        if "TESSDATA_PREFIX" not in env:
            for candidate in (
                bundled_tessdata,                       # project-local (preferred)
                Path("/opt/homebrew/share/tessdata"),   # Apple-silicon brew
                Path("/usr/local/share/tessdata"),      # Intel brew
                Path("/usr/share/tesseract-ocr/4.00/tessdata"),  # Debian/Ubuntu
                Path("/usr/share/tesseract-ocr/5/tessdata"),
                Path("/usr/share/tessdata"),            # generic
            ):
                if candidate.is_dir() and (candidate / "eng.traineddata").is_file():
                    env["TESSDATA_PREFIX"] = str(candidate)
                    break

        # `-constant` flags override Audiveris's internal defaults.
        # Profiles.defaultQuality tightens (Synthetic) or loosens (Poor)
        # recognition thresholds based on input source. The OCR DPI hint
        # tells Tesseract the actual resolution of the image it's reading
        # — without this it auto-estimates, which is unreliable for the
        # 300 DPI synthetic renders we produce client-side. Better hint
        # → better tempo/dynamics text accuracy.
        ocr_dpi = 300 if quality == "Synthetic" else 200
        constant_args = [
            "-constant",
            f"org.audiveris.omr.sheet.Profiles.defaultQuality={quality}",
            "-constant",
            f"org.audiveris.omr.text.tesseract.TesseractOrder.typicalImageResolution={ocr_dpi}",
        ]

        try:
            result = subprocess.run(
                [
                    "audiveris",
                    "-batch",          # headless, no GUI
                    "-export",         # generate MusicXML
                    "-sheets", "1",    # only first sheet — Audiveris auto-splits
                                       # tall images and a bogus second "sheet"
                                       # will fail validation and abort the export
                    *constant_args,
                    "-output", str(workdir),
                    str(img_path),
                ],
                capture_output=True,
                text=True,
                timeout=AUDIVERIS_TIMEOUT_SECONDS,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise OmrError(
                f"OMR timed out after {AUDIVERIS_TIMEOUT_SECONDS}s"
            ) from exc
        except FileNotFoundError as exc:
            raise OmrError(
                "`audiveris` CLI not found on PATH — is it installed in this environment?"
            ) from exc

        if result.returncode != 0:
            combined = (result.stdout or "") + "\n" + (result.stderr or "")
            raise OmrError(_friendly_audiveris_error(combined, result.returncode))

        # Audiveris writes results into a subdirectory named after the input
        # file (e.g. workdir/input/input.mxl). Walk to find them.
        candidates = list(workdir.rglob("*.mxl")) + list(workdir.rglob("*.musicxml"))
        if not candidates:
            raise OmrError("Audiveris produced no MusicXML output")
        # Prefer .mxl (compressed) when both exist — that's what the
        # MusicXML parser on the client already handles.
        candidates.sort(key=lambda p: 0 if p.suffix == ".mxl" else 1)
        return workdir, candidates[0]
    except BaseException:
        # Any error path (raised OmrError, KeyboardInterrupt, …) must not
        # leave the workdir on disk; only the success path hands ownership
        # to the caller.
        shutil.rmtree(workdir, ignore_errors=True)
        raise


def _preprocess(image_bytes: bytes) -> bytes:
    """Clean the image up before handing it to Audiveris. Audiveris is
    sensitive to lighting, blur, and rotation; a few cheap operations
    measurably improve its hit rate on real-world phone photos.

    Pipeline (each step is a no-op on already-clean inputs):
      1. Decode + force to grayscale "L" — Audiveris ignores colour and
         a grayscale image lets autocontrast operate on a single channel.
      2. Fail-fast on essentially-blank inputs (no ink) — saves a
         pointless 30s Audiveris timeout on accidental uploads.
      3. Auto-crop to the bbox of the actual content with a small margin
         — phone photos almost always include desk / wall / shadows
         around the music, and Audiveris's binarization can latch onto
         that noise instead of the staff lines.
      4. Autocontrast — stretches the histogram so faded scans / dim
         lighting become high-contrast black-on-white.
      5. UnsharpMask — counters mild phone-camera focus blur.
      6. Deskew — detects the rotation that gives staff lines their
         strongest horizontal projection and rotates the image flat.
      7. Upscale-if-small — ensures Audiveris has enough pixels per
         staff line.

    On any decode error the original bytes are returned unchanged so a
    malformed-but-valid image still gets a chance at recognition.
    """
    try:
        im = Image.open(io.BytesIO(image_bytes))
        im.load()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        # A maliciously-crafted image whose pixel count would balloon past
        # Pillow's safety threshold (or half of it, via the warning we
        # promote to an exception at module load). Refuse rather than
        # handing the raw bytes to Audiveris, which would OOM the JVM.
        raise OmrError(
            "Image is too large to process safely; please use a smaller scan."
        ) from exc
    except Exception:  # noqa: BLE001
        return image_bytes

    # Reject pathologically small images — sheet music below this size has
    # no chance of OMR working, and downstream numpy ops on a 0-sized array
    # would crash.
    if min(im.size) < 50:
        raise OmrError(
            "Image too small for note recognition. "
            "Use a larger photo or screenshot."
        )

    # Force to single-channel grayscale.
    if im.mode != "L":
        im = im.convert("L")

    # Fail fast on blank uploads (no ink at all) — saves Audiveris the
    # 30 s timeout dance and gives the user a clearer error.
    _check_not_blank(im)

    # Tight content bbox + small margin — removes background noise that
    # confuses Audiveris's binarization step. No-op on PDF renders that
    # are already tight.
    im = _auto_crop(im)

    # 1px cutoff at each end of the histogram so an outlier dark pixel
    # (sensor dust, watermark) doesn't peg the contrast.
    im = ImageOps.autocontrast(im, cutoff=1)

    # Three-pass sharpening, each pass targets a different feature scale:
    #   - Pass 1 (radius 1.2): general edge enhancement for staff lines and
    #     individual noteheads. Too aggressive a percent here introduces
    #     ringing Audiveris mistakes for staff lines.
    #   - Pass 2 (radius 0.6): targets fine details — accidentals
    #     (♯ ♭ ♮), dots, articulations. These are the symbols Audiveris
    #     most often misreads and are small enough that pass-1's wider
    #     kernel barely affects them.
    #   - Pass 3 (radius 0.4): ultra-fine, aimed at the inter-notehead
    #     gap inside stacked chord noteheads. When two notes sit a third
    #     apart on adjacent lines/spaces, Audiveris frequently fuses them
    #     into one notehead. A tight high-pass restores the separation.
    #     Low percent + higher threshold keeps it from adding noise.
    im = im.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=2))
    im = im.filter(ImageFilter.UnsharpMask(radius=0.6, percent=160, threshold=3))
    im = im.filter(ImageFilter.UnsharpMask(radius=0.4, percent=110, threshold=4))

    # Deskew: only correct rotations > 0.5°; smaller is within Audiveris's
    # tolerance and re-rotating would just blur the image for no benefit.
    angle = _detect_skew_angle(im)
    if abs(angle) > 0.5:
        log.info("deskewing by %.2f°", -angle)
        im = im.rotate(-angle, resample=Image.BICUBIC, fillcolor=255,
                        expand=False)

    # Upscale-if-small, same as before.
    longest = max(im.size)
    if longest < MIN_LONGEST_EDGE:
        scale = MIN_LONGEST_EDGE / longest
        new_size = (round(im.width * scale), round(im.height * scale))
        log.info("upscaling %s → %s (×%.2f) for Audiveris",
                 im.size, new_size, scale)
        im = im.resize(new_size, Image.BICUBIC)

    out = io.BytesIO()
    im.save(out, format="PNG")
    return out.getvalue()


def _preprocess_minimal(image_bytes: bytes) -> bytes:
    """Lightweight cleanup for already-clean inputs (PDF rasterizations).
    Skips autocontrast and the unsharp-mask stack — those are noise-fighting
    steps that introduce ringing/halos on perfect inputs (and were hurting
    chord-notehead separation on dense scores). We still:
      - Force grayscale (Audiveris ignores colour).
      - Deskew. Beams are near-horizontal strokes whose detection collapses
        on even sub-degree rotation; skipping this step measurably degraded
        rhythm parsing (eighth-note beam groups misread as isolated quarter
        notes). Cheap, near-no-op on already-square inputs.
      - Upscale-if-small to keep staff lines above Audiveris's 15 px/interline
        floor (rare for PDF renders, but cheap insurance).
    """
    try:
        im = Image.open(io.BytesIO(image_bytes))
        im.load()
    except Image.DecompressionBombError as exc:
        raise OmrError(
            "Image is too large to process safely; please use a smaller scan."
        ) from exc
    except Exception:  # noqa: BLE001
        return image_bytes

    if min(im.size) < 50:
        raise OmrError(
            "Image too small for note recognition. "
            "Use a larger photo or screenshot."
        )

    if im.mode != "L":
        im = im.convert("L")

    # Fail fast on blank inputs even in the synthetic path (the user may
    # have accidentally uploaded a PDF cover page or a blank first page).
    _check_not_blank(im)

    # Deskew using the same row-projection-variance estimator as the full
    # pipeline. Only correct >0.5°; smaller is within Audiveris's tolerance
    # and re-rotating would blur the image for no benefit.
    angle = _detect_skew_angle(im)
    if abs(angle) > 0.5:
        log.info("deskewing by %.2f° (synthetic path)", -angle)
        im = im.rotate(-angle, resample=Image.BICUBIC, fillcolor=255,
                        expand=False)

    longest = max(im.size)
    if longest < MIN_LONGEST_EDGE:
        scale = MIN_LONGEST_EDGE / longest
        new_size = (round(im.width * scale), round(im.height * scale))
        log.info("upscaling %s → %s (×%.2f) for Audiveris",
                 im.size, new_size, scale)
        im = im.resize(new_size, Image.BICUBIC)

    out = io.BytesIO()
    im.save(out, format="PNG")
    return out.getvalue()


def _estimate_sharpness(image_bytes: bytes) -> float | None:
    """Laplacian-variance blur metric. Higher = sharper edges, lower =
    blurrier. Operates on a downscaled grayscale thumbnail so the cost
    is bounded regardless of input resolution.

    Returns None on decode failure so the caller can choose its own
    fallback behavior. Typical values: a clean PDF render scores
    ~1200–3000; a 300 DPI photo of well-lit sheet music ~600–1200; an
    upscaled 75 DPI scan-in-PDF scores ~50–150.
    """
    try:
        im = Image.open(io.BytesIO(image_bytes))
        im.load()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        # Sharpness is diagnostic; if the image trips the bomb thresholds
        # the real preprocess path will reject it with a friendlier error.
        return None
    except Exception:  # noqa: BLE001
        return None
    if im.mode != "L":
        im = im.convert("L")
    # 600 px thumbnail keeps the convolution under a few ms while
    # preserving enough high-frequency content for the variance to be
    # meaningful. Don't go below ~400 — at that point the downsample
    # itself blurs edges and the metric collapses.
    im.thumbnail((600, 600), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32)
    # 4-neighbor Laplacian (no SciPy dep): center − sum-of-neighbors.
    lap = (
        arr[1:-1, 1:-1] * 4.0
        - arr[:-2, 1:-1] - arr[2:, 1:-1]
        - arr[1:-1, :-2] - arr[1:-1, 2:]
    )
    return float(lap.var())


def _check_not_blank(im: Image.Image) -> None:
    """Raise OmrError if the image has no ink content to speak of. Lets us
    fail in under a second instead of waiting 30+ s for Audiveris to
    declare it can't find a staff. Threshold tuned so that even a
    very-lightly-printed score (low-contrast scan) passes; only true
    blanks / solid-color uploads trip this.

    Operates on a 256-pixel thumbnail so it's microseconds regardless of
    input size.
    """
    thumb = im.copy()
    thumb.thumbnail((256, 256), Image.BILINEAR)
    arr = np.asarray(thumb, dtype=np.uint8)
    # "Ink" = pixels meaningfully darker than mid-gray. Threshold 200
    # keeps light printing in (200 is well above typical scan grays for
    # printed staves).
    ink_ratio = float((arr < 200).mean())
    if ink_ratio < 0.005:  # less than 0.5% ink → blank
        raise OmrError(
            "Image appears blank — no sheet music content detected. "
            "Make sure the photo or PDF shows printed staves and is not "
            "a cover page or solid-colour image."
        )


def _auto_crop(im: Image.Image) -> Image.Image:
    """Trim the image to the bounding box of the actual content, plus a
    small margin. The biggest win is on phone photos that include desk /
    wall / shadows around the score — Audiveris's binarization step can
    latch onto the background gradient and miss the staves. On a clean
    PDF render this is a near no-op because the page is already tight.

    Conservative: refuses to crop unless there's a clear content/background
    contrast (≥1% ink) and the proposed crop would actually remove
    something (≥5% area saved). Otherwise returns the input unchanged.
    """
    arr = np.asarray(im, dtype=np.uint8)
    # 240 picks up everything but bright background. Higher than the
    # blank-check threshold so smudged scan backgrounds don't shift the
    # bbox outward.
    mask = arr < 240
    if mask.mean() < 0.01:
        return im

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return im
    top, bottom = int(rows[0]), int(rows[-1])
    left, right = int(cols[0]), int(cols[-1])

    h, w = arr.shape[:2]
    # Cheap heuristic: if the bbox already fills most of the page, the
    # margins are negligible and cropping just adds the risk of cutting
    # off a high notehead / dynamic marking. Bail out.
    bbox_area = (bottom - top + 1) * (right - left + 1)
    if bbox_area >= 0.95 * h * w:
        return im

    # 2% margin on each side so we don't cut staff-line tails or ledger
    # noteheads sitting right at the edge of the content.
    margin_h = max(8, int(h * 0.02))
    margin_w = max(8, int(w * 0.02))
    top = max(0, top - margin_h)
    bottom = min(h - 1, bottom + margin_h)
    left = max(0, left - margin_w)
    right = min(w - 1, right + margin_w)

    cropped = im.crop((left, top, right + 1, bottom + 1))
    log.info(
        "auto-crop %dx%d → %dx%d (%.0f%% of original)",
        w, h, cropped.size[0], cropped.size[1],
        100.0 * cropped.size[0] * cropped.size[1] / (w * h),
    )
    return cropped


def _detect_skew_angle(im: Image.Image) -> float:
    """Find the rotation that maximises the variance of horizontal row-sums
    on an inverted (ink-bright) thumbnail. Sheet music has strong horizontal
    staff lines that line up to a sharp row-sum peak when the image is
    square; off-axis they smear out.

    Two-stage search: a coarse pass over a wider range catches handheld-scan
    rotations up to ±8°, then a fine pass narrows to 0.1° precision around
    the coarse winner. Wider+finer than the previous ±5° / 0.5° because
    real scans (and especially photos held in landscape vs portrait) can
    arrive 6–8° off, and sub-half-degree precision measurably tightens
    beam detection.
    """
    # Downscale for the search — 600 px on the longest edge keeps each
    # candidate rotation under a few milliseconds.
    w, h = im.size
    longest = max(w, h)
    scale = 600 / longest if longest > 600 else 1.0
    thumb = (im.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
             if scale < 1 else im)

    def score_angle(angle: float) -> float:
        rotated = thumb.rotate(float(angle), resample=Image.BILINEAR,
                                fillcolor=255, expand=False)
        arr = 255 - np.asarray(rotated, dtype=np.float32)
        row_sums = arr.sum(axis=1)
        return float(row_sums.var())

    # Coarse pass: ±8° in 0.5° steps.
    best_angle = 0.0
    best_score = -1.0
    for angle in np.arange(-8.0, 8.01, 0.5):
        s = score_angle(float(angle))
        if s > best_score:
            best_score = s
            best_angle = float(angle)

    # Fine pass: ±0.5° around the coarse winner in 0.1° steps.
    for angle in np.arange(best_angle - 0.5, best_angle + 0.51, 0.1):
        s = score_angle(float(angle))
        if s > best_score:
            best_score = s
            best_angle = float(angle)

    return best_angle


def _friendly_audiveris_error(output: str, exit_code: int) -> str:
    """Translate Audiveris's verbose CLI output into a one-line message the
    Flutter client can show. The raw CLI log is kept server-side only —
    it can contain filesystem paths, JVM internals, and other detail that
    is useful for debugging but not safe to return to a remote caller."""
    if "too low interline" in output or "picture resolution is too low" in output:
        return (
            "Image resolution too low for note recognition. "
            "Use a sharper photo (~300 DPI / 1500+ pixels tall) or a screenshot "
            "of the score at its original size."
        )
    if "no multi-line staves" in output or "flagged as invalid" in output:
        return (
            "Couldn't find staff lines in this image. Make sure the photo "
            "shows printed sheet music with the whole staff visible and "
            "isn't cropped, rotated, or skewed."
        )
    if "Error in export" in output:
        return (
            "Audiveris ran but couldn't transcribe this score. "
            "Try a clearer image or simpler passage."
        )
    log.warning("audiveris exit=%d, tail=%r",
                exit_code, output.strip().splitlines()[-5:])
    return "Sheet music recognition failed. Try a clearer image."
