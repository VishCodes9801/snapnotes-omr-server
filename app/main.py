"""FastAPI entrypoint for the SnapNotes OMR service.

Endpoints:
  GET  /health          → liveness probe
  POST /extract         → multipart image upload, returns {bpm, notes}
"""

import asyncio
import hashlib
import io
import json
import logging
import os
import re
import secrets
import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse

import base64

from . import cache
from .convert import musicxml_to_payload
from .layout import detect_system_bands
from .omr import PIPELINE_VERSION, VALID_QUALITIES, OmrError, run_audiveris

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("snapnotes")

# Per-file upload cap. A page rasterized at 2400 px on the longest edge
# usually weighs in around 1–3 MB; 50 MB is generous headroom for very
# detailed scans while still rejecting a runaway client trying to OOM us.
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# Per-request cap on the *number* of pages. A single /extract call runs
# Audiveris serially per page (each ~240 s timeout, 2 GB JVM heap), so an
# unbounded list lets one request pin the worker for hours. Real piano
# scores in our test corpus top out around 8 pages; 20 is generous
# headroom without permitting a DoS via giant lists.
MAX_PAGES_PER_REQUEST = 20

# Per-request cap on total bytes summed across pages. Per-file cap alone
# wouldn't stop 20 files × 50 MB = 1 GB of memory pressure during the
# read loop.
MAX_TOTAL_UPLOAD_BYTES = 200 * 1024 * 1024

# Magic-byte prefixes for the image formats we actually accept. Rejecting
# at the HTTP edge keeps untrusted bytes out of Pillow's decoders entirely
# — a useful defence-in-depth layer the day the next Pillow CVE drops.
_ALLOWED_IMAGE_MAGIC: tuple[bytes, ...] = (
    b"\x89PNG\r\n\x1a\n",  # PNG
    b"\xff\xd8\xff",        # JPEG
)


# Optional shared-secret auth for /extract. Empty (the default) leaves the
# endpoint unauthenticated — fine for localhost / LAN where the rate
# limiter is the only gate. Public deployments should set this so a single
# attacker can't sustain Audiveris workers across IPs. Clients send the
# value as `Authorization: Bearer <token>`.
_API_TOKEN = os.getenv("SNAPNOTES_API_TOKEN", "").strip()


def _check_auth(request: Request) -> None:
    """Constant-time bearer-token check. No-op when SNAPNOTES_API_TOKEN is
    unset, so localhost/LAN setups keep working without configuration."""
    if not _API_TOKEN:
        return
    header = request.headers.get("authorization", "")
    prefix = "Bearer "
    if not header.startswith(prefix) or not secrets.compare_digest(
        header[len(prefix):], _API_TOKEN
    ):
        raise HTTPException(status_code=401, detail="unauthorized")


def _allowed_origins() -> list[str]:
    """Read CORS origins from ALLOWED_ORIGINS env var (comma-separated).
    Empty default = no cross-origin browser callers, which is what the
    Flutter app needs (it talks server-to-device, not browser-to-server).
    Set ALLOWED_ORIGINS in deployment to enable specific web frontends."""
    raw = os.getenv("ALLOWED_ORIGINS", "").strip()
    if not raw:
        return []
    return [o.strip() for o in raw.split(",") if o.strip()]


limiter = Limiter(key_func=get_remote_address)

# One OMR job at a time, enforced in-process. Each Audiveris run is a
# ~2 GB-heap JVM; two overlapping /extract calls on one instance would
# OOM-kill both. The endpoint's blocking subprocess call already
# serializes today's single-worker deployment, but this lock keeps that
# guarantee explicit and survives any future move to threaded handlers.
# Pair with a platform-level concurrency limit of 1 per instance.
_scan_lock = asyncio.Lock()

app = FastAPI(title="SnapNotes OMR")
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(_request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": f"rate limit exceeded: {exc.detail}"},
    )


# Restrictive CORS by default; deployments that need browser callers from
# specific origins set ALLOWED_ORIGINS=https://app.example.com,...
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ── Anonymous usage tallies ──────────────────────────────────────────────────
#
# The app reports bare event names ("scan_succeeded"); we keep one integer
# per name. No identifiers, no per-user rows, no timestamps beyond the log
# line — the data can answer "how many scans succeeded" and nothing about
# any individual. Storage is a JSON file next to the OMR cache, which on
# Fly persists across machine stop/start but resets on redeploy; every
# event is also logged, so history is recoverable from `fly logs` if it
# ever matters.

_EVENT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
_MAX_EVENTS_PER_REQUEST = 20
_MAX_COUNT_PER_EVENT = 100

_STATS_PATH = Path(
    os.getenv("SNAPNOTES_STATS_PATH", "cache/usage_stats.json")
)
_stats_lock = asyncio.Lock()


def _load_stats() -> dict[str, int]:
    try:
        raw = json.loads(_STATS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        k: int(v)
        for k, v in raw.items()
        if isinstance(k, str) and isinstance(v, (int, float))
    }


@app.post("/events")
@limiter.limit("30/minute")
async def events(request: Request) -> dict:
    _check_auth(request)
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 — malformed JSON of any flavour
        raise HTTPException(status_code=400, detail="invalid JSON body")
    counts = body.get("counts") if isinstance(body, dict) else None
    if not isinstance(counts, dict) or not counts:
        raise HTTPException(status_code=400, detail="missing counts")
    if len(counts) > _MAX_EVENTS_PER_REQUEST:
        raise HTTPException(status_code=400, detail="too many events")
    cleaned: dict[str, int] = {}
    for name, n in counts.items():
        if not isinstance(name, str) or not _EVENT_NAME_RE.match(name):
            raise HTTPException(
                status_code=400, detail=f"bad event name: {name!r}"
            )
        if not isinstance(n, int) or not 1 <= n <= _MAX_COUNT_PER_EVENT:
            raise HTTPException(
                status_code=400, detail=f"bad count for {name}"
            )
        cleaned[name] = n

    async with _stats_lock:
        stats = _load_stats()
        for name, n in cleaned.items():
            stats[name] = stats.get(name, 0) + n
        try:
            _STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
            _STATS_PATH.write_text(json.dumps(stats))
        except OSError:
            log.warning("stats write failed (non-fatal)", exc_info=True)
    log.info("events: %s", cleaned)
    return {"ok": True}


@app.get("/stats")
async def stats(request: Request) -> dict:
    """Aggregate counters, for the operator (token-gated like /extract)."""
    _check_auth(request)
    async with _stats_lock:
        return {"counts": _load_stats()}


@app.post("/extract")
@limiter.limit("5/minute")
async def extract(
    request: Request,
    images: list[UploadFile] = File(...),
    force: bool = False,
    quality: str = "Standard",
    include_pages: bool = False,
) -> dict:
    _check_auth(request)

    if not images:
        raise HTTPException(status_code=400, detail="no images uploaded")

    if len(images) > MAX_PAGES_PER_REQUEST:
        raise HTTPException(
            status_code=413,
            detail=(
                f"too many pages: {len(images)} "
                f"(max {MAX_PAGES_PER_REQUEST} per request)"
            ),
        )

    if quality not in VALID_QUALITIES:
        raise HTTPException(
            status_code=400,
            detail=f"quality must be one of {VALID_QUALITIES}",
        )

    raws: list[bytes] = []
    total_bytes = 0
    # Build the cache hash incrementally as bytes arrive so we never have
    # to allocate a single buffer the size of every page concatenated
    # (which could be up to MAX_TOTAL_UPLOAD_BYTES = 200 MB).
    hasher = hashlib.sha256()
    # include_pages is part of the key: the payloads differ (score-view
    # page images + system geometry vs notes only).
    hasher.update(
        f"{PIPELINE_VERSION}|{quality}|pages={int(include_pages)}|".encode()
    )
    for img in images:
        raw = await img.read()
        if not raw:
            raise HTTPException(
                status_code=400, detail="empty image upload"
            )
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"page is {len(raw) // (1024 * 1024)} MB; "
                    f"max is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB per page."
                ),
            )
        total_bytes += len(raw)
        if total_bytes > MAX_TOTAL_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"total upload exceeds "
                    f"{MAX_TOTAL_UPLOAD_BYTES // (1024 * 1024)} MB"
                ),
            )
        if not raw.startswith(_ALLOWED_IMAGE_MAGIC):
            raise HTTPException(
                status_code=415,
                detail="unsupported image format (PNG or JPEG only)",
            )
        hasher.update(raw)
        raws.append(raw)
    cache_key = hasher.hexdigest()
    # `force` lets the client re-scan the same image bytes without getting
    # a stale cached payload — needed when preprocessing changes or the
    # user just wants a fresh OMR pass. We still write the new result to
    # cache so the next normal call hits it.
    if not force:
        cached = cache.get(cache_key)
        if cached is not None:
            log.info(
                "cache hit (%d pages, %d bytes)", len(raws), total_bytes
            )
            return cached
    else:
        log.info("force re-scan (%d pages, %d bytes)", len(raws), total_bytes)

    log.info("running OMR on %d page(s)", len(raws))

    merged_notes: list[dict] = []
    bpm: float | None = None
    offset = 0.0
    page_boundaries: list[float] = []  # offset at the start of each page > 0
    page_metas: list[dict] = []
    score_pages: list[dict] = []
    # Track the most recently-seen time signature so later pages whose
    # MusicXML lacks one inherit it directly into the music21 score
    # (and therefore through drift compensation). Without this, pages
    # 2+ of a multi-page PDF skip the rhythm anchoring entirely.
    known_time_signature: str | None = None

    for i, raw in enumerate(raws, start=1):
        workdir = None
        try:
            async with _scan_lock:
                workdir, xml_path = run_audiveris(raw, quality=quality)
            page = musicxml_to_payload(
                xml_path,
                fallback_time_signature=known_time_signature,
            )
            if include_pages:
                # The preprocessed PNG is the image Audiveris actually
                # read, so its pixels and the detected geometry always
                # agree — regardless of what preprocessing did to the
                # upload.
                score_pages.append(
                    _build_score_page(workdir / "input.png", page, offset)
                )
        except OmrError as exc:
            # OmrError messages are intentionally user-facing (e.g.
            # "Image resolution too low"); safe to forward to the client.
            log.warning("OMR failed on page %d: %s", i, exc)
            raise HTTPException(
                status_code=422, detail=f"page {i}: {exc}"
            ) from exc
        except Exception:  # noqa: BLE001
            # Don't leak raw exception strings — they can contain
            # filesystem paths, library internals, JVM stack traces.
            # The detail is logged server-side; the client gets a
            # generic message.
            log.exception("unexpected OMR failure on page %d", i)
            raise HTTPException(
                status_code=500,
                detail=f"page {i}: internal server error",
            )
        finally:
            if workdir is not None:
                shutil.rmtree(workdir, ignore_errors=True)

        if bpm is None:
            bpm = float(page.get("bpm", 90.0))

        if i > 1:
            page_boundaries.append(offset)

        for note in page.get("notes", []):
            merged_notes.append({**note, "beat": float(note["beat"]) + offset})

        if page.get("meta"):
            page_metas.append(page["meta"])
            # Remember this page's time signature for inheriting into
            # any later page that lacks one in its own MusicXML.
            ts = page["meta"].get("time_signature")
            if ts:
                known_time_signature = ts

        # Prefer music21's score duration (includes trailing rests / bars);
        # fall back to the end of the last note if it's missing.
        page_beats = float(page.get("total_beats") or 0.0)
        if page_beats <= 0 and page.get("notes"):
            page_beats = max(
                float(n["beat"]) + float(n["dur"]) for n in page["notes"]
            )
        offset += page_beats

    # Cross-page tie merging. music21's stripTies runs per-page, so a tie
    # that crosses a page boundary leaves us with two adjacent notes
    # (same pitch, same hand, second starts where first ends) at the
    # boundary offset. Detect that shape and merge.
    merges = _merge_cross_page_ties(merged_notes, page_boundaries)
    if merges:
        log.info("cross-page tie merges: %d", merges)

    # Inherit missing key / time signature from prior pages. Audiveris
    # sometimes drops the key glyph on dense first systems of later
    # pages; carrying forward stops the visualizer from re-spelling
    # everything for half the score.
    _inherit_page_meta(page_metas)

    # Clamp output to sane ranges so a single Audiveris glyph
    # misclassification (e.g., a notehead read as MIDI 200, or a
    # whole-rest duration of 999) doesn't reach the visualizer.
    sanitized = _sanitize_notes(merged_notes)

    sanitized.sort(key=lambda e: (e["beat"], e["note"]))
    payload: dict = {
        "bpm": bpm if bpm is not None else 90.0,
        "notes": sanitized,
    }
    if include_pages:
        payload["score_pages"] = score_pages
    if page_metas:
        payload["meta"] = {
            "pages": page_metas,
            "cross_page_tie_merges": merges,
            "quality": quality,
        }
    # Cache write is best-effort: a read-only or full disk must not turn a
    # successful multi-minute scan into a 500.
    try:
        cache.put(cache_key, payload)
    except OSError:
        log.warning("cache write failed (non-fatal)", exc_info=True)
    log.info(
        "extracted %d note events across %d page(s)",
        len(merged_notes),
        len(raws),
    )
    return payload


# Score-view page images: bounded width keeps a page around 60–150 KB
# as a JPEG — enough resolution to read the notation on a phone panel.
_SCORE_PAGE_MAX_WIDTH = 1100
_SCORE_PAGE_JPEG_QUALITY = 70


def _build_score_page(png_path, page: dict, page_offset: float) -> dict:
    """Assemble one score-view page: the preprocessed image (downscaled
    JPEG, base64) plus normalized system bands with their start beats.
    System geometry comes from the image (projection profile) and the
    system *membership* from MusicXML layout breaks; when the two
    disagree the page ships without systems and the client falls back to
    page-level sync."""
    from PIL import Image  # local import keeps module import cheap

    png_bytes = png_path.read_bytes()
    im = Image.open(io.BytesIO(png_bytes))
    im.load()
    if im.mode != "L":
        im = im.convert("L")
    if im.width > _SCORE_PAGE_MAX_WIDTH:
        im = im.resize(
            (
                _SCORE_PAGE_MAX_WIDTH,
                max(1, round(im.height * _SCORE_PAGE_MAX_WIDTH / im.width)),
            ),
            Image.BILINEAR,
        )
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=_SCORE_PAGE_JPEG_QUALITY)
    image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    staves_per_system = int(page.get("meta", {}).get("parts") or 2)
    bands = detect_system_bands(png_bytes, staves_per_system)
    n_sys_xml = int(page.get("system_count") or 1)
    measures = page.get("measures") or []

    systems: list[dict] = []
    if bands and len(bands) == n_sys_xml and measures:
        # Chronological measure beats per system; a repeat that revisits
        # a line contributes its beats to that line again, which is what
        # the client's playhead wants.
        by_system: dict[int, list[float]] = {}
        for m in measures:
            by_system.setdefault(int(m["system"]), []).append(
                float(m["beat"]) + page_offset
            )
        if all(s in by_system for s in range(len(bands))):
            systems = [
                {
                    **band,
                    "beat": min(by_system[idx]),
                    "measures": sorted(by_system[idx]),
                }
                for idx, band in enumerate(bands)
            ]
    if not systems:
        log.info(
            "score page: system fallback (image bands=%d, xml systems=%d)",
            len(bands),
            n_sys_xml,
        )
    return {
        "image": image_b64,
        "start_beat": page_offset,
        "systems": systems,
    }


_PIANO_MIDI_LOW = 21    # A0
_PIANO_MIDI_HIGH = 108  # C8
_MAX_NOTE_DURATION_BEATS = 32.0  # 8 whole notes — generous upper bound
_NOTE_NAME_TO_PITCH_CLASS = {
    "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11,
}


def _midi_from_note_name(name: str) -> int | None:
    """Parse the scientific-pitch string convert.py produces (e.g. 'F#4',
    'Bb-1', 'C##3') and return the MIDI number. None for malformed input."""
    if not name:
        return None
    base = name[0]
    if base not in _NOTE_NAME_TO_PITCH_CLASS:
        return None
    pc = _NOTE_NAME_TO_PITCH_CLASS[base]
    idx = 1
    while idx < len(name) and name[idx] in ("#", "b"):
        pc += 1 if name[idx] == "#" else -1
        idx += 1
    try:
        octave = int(name[idx:])
    except (ValueError, IndexError):
        return None
    return (octave + 1) * 12 + pc


def _sanitize_notes(notes: list[dict]) -> list[dict]:
    """Drop / clamp obviously-bogus notes that would crash or visually break
    the visualizer. Audiveris very occasionally emits absurd values when
    a glyph misclassification cascades through duration / pitch
    inference; this catches those without affecting the well-formed
    majority of notes."""
    out: list[dict] = []
    dropped = {"unparseable": 0, "out_of_range": 0, "bad_duration": 0,
               "negative_beat": 0}
    for n in notes:
        midi = _midi_from_note_name(n.get("note", ""))
        if midi is None:
            dropped["unparseable"] += 1
            continue
        if midi < _PIANO_MIDI_LOW or midi > _PIANO_MIDI_HIGH:
            dropped["out_of_range"] += 1
            continue
        beat = float(n.get("beat", 0.0))
        dur = float(n.get("dur", 0.0))
        if dur <= 0 or dur > _MAX_NOTE_DURATION_BEATS:
            dropped["bad_duration"] += 1
            continue
        if beat < 0:
            # Shouldn't happen after offset accumulation; if it does,
            # the note is in an inconsistent state. Skip rather than
            # render it before the piece starts.
            dropped["negative_beat"] += 1
            continue
        out.append(n)
    total_dropped = sum(dropped.values())
    if total_dropped:
        log.info("sanitize dropped %d note(s): %s", total_dropped, dropped)
    return out


def _inherit_page_meta(page_metas: list[dict]) -> None:
    """Fill in missing key / time signature on later pages by carrying
    forward the most recent value from a prior page. Mutates the metas
    in place. Music doesn't change key per page — but Audiveris drops
    the key glyph on dense first systems of later pages often enough
    that this is a real win.

    Adds an `inherited` list per page noting which fields were carried
    forward, so the diagnostic stays honest (you can see when the value
    came from this page vs from upstream)."""
    carry: dict[str, Any] = {}
    for meta in page_metas:
        inherited = []
        for field in ("key", "time_signature"):
            if meta.get(field):
                carry[field] = meta[field]
            elif field in carry:
                meta[field] = carry[field]
                inherited.append(field)
        if inherited:
            meta["inherited"] = inherited


def _merge_cross_page_ties(
    notes: list[dict], page_boundaries: list[float]
) -> int:
    """Merge same-pitch + same-hand notes whose endpoints meet exactly at a
    page boundary — they were almost certainly a single tied note that
    music21's per-page stripTies couldn't see across files.

    Returns the count of merges performed. Mutates `notes` in place by
    extending the first note's `dur` and removing the second.

    Heuristic, with a small float tolerance for the boundary equality
    check. Conservative: we only merge when (a) the second note starts
    AT the boundary, (b) the first note ends AT the boundary, (c) same
    pitch string, (d) same hand string. The combination of constraints
    makes false positives rare in practice — same-pitch re-articulation
    at the exact frame of a page break is uncommon enough that merging
    it is the right call almost always.
    """
    if not page_boundaries or not notes:
        return 0

    boundary_set = {round(b, 4) for b in page_boundaries}
    eps = 1e-3

    # Index notes by their (start_beat, note, hand) for O(1) lookup of
    # "is there a note starting at this boundary that matches mine".
    starts_at: dict[tuple[float, str, str], int] = {}
    for idx, n in enumerate(notes):
        key = (round(float(n["beat"]), 4), n.get("note"), n.get("hand"))
        starts_at.setdefault(key, idx)

    merged_count = 0
    to_drop: set[int] = set()
    for idx, n in enumerate(notes):
        if idx in to_drop:
            continue
        end = float(n["beat"]) + float(n["dur"])
        end_rounded = round(end, 4)
        if end_rounded not in boundary_set:
            continue
        key = (end_rounded, n.get("note"), n.get("hand"))
        other = starts_at.get(key)
        if other is None or other == idx or other in to_drop:
            continue
        # Sanity: only merge if the floats really align within tolerance.
        if abs(notes[other]["beat"] - end) > eps:
            continue
        n["dur"] = float(n["dur"]) + float(notes[other]["dur"])
        to_drop.add(other)
        merged_count += 1

    if to_drop:
        notes[:] = [n for i, n in enumerate(notes) if i not in to_drop]
    return merged_count
