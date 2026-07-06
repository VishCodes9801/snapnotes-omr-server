"""Converts a MusicXML file (Audiveris's output) to the NoteEvent JSON shape
the Flutter app already consumes:

  {
    "bpm": 120.0,
    "notes": [
      {"note": "D4", "beat": 0.0, "dur": 1.0, "hand": "right", "vel": 80},
      ...
    ]
  }

Beats are quarter-notes, matching music21's default offset/duration units."""

import logging
from pathlib import Path
from typing import Any

from bisect import bisect_right

import music21
from music21 import dynamics as m21_dynamics
from music21 import expressions as m21_expressions
from music21 import key as m21_key
from music21 import layout as m21_layout
from music21 import meter as m21_meter
from music21 import stream as m21_stream
from music21 import tempo as m21_tempo

log = logging.getLogger("snapnotes.convert")


# Mapping from MusicXML dynamic markings to MIDI velocity. Standard piano
# practice with mid-range values that read clearly in playback without
# clipping at either end. Sources: General MIDI conventions + music21's
# `dynamics.dynamicStrToScalar` (we hardcode rather than depend on the
# private helper since values shift slightly between music21 versions).
_DYNAMIC_VELOCITY = {
    "ppp": 16,
    "pp": 33,
    "p": 49,
    "mp": 64,
    "mf": 80,
    "f": 96,
    "ff": 112,
    "fff": 127,
    # Accent-style markings — sharper attack, not sustained loudness.
    "sf": 112,
    "sfz": 120,
    "fp": 96,    # forte attack, immediate piano
    "rfz": 112,
    "sfp": 112,
}

# Articulation modifiers as (velocity multiplier, duration multiplier),
# applied on top of the dynamic-derived base velocity. Conservative values
# that read clearly without sounding caricatured.
_ARTICULATION_TABLE = {
    "Staccato":      (1.00, 0.50),
    "Staccatissimo": (1.00, 0.25),
    "Tenuto":        (1.00, 1.00),
    "Accent":        (1.25, 1.00),
    "StrongAccent":  (1.40, 1.00),  # marcato
    "Fermata":       (1.00, 2.00),
}

# Default velocity when no dynamic has been seen yet — "mezzo" feel.
_DEFAULT_VELOCITY = 80
# Default BPM when neither a MetronomeMark nor a parseable TempoText is
# found. Matches the Flutter client's AppConstants.defaultBpm so the
# "no tempo info" path lands at the same value everywhere.
_DEFAULT_BPM = 90.0
# Grace notes often arrive from music21 with a tiny duration (or 0). The
# visualizer can't render them at that size — promote anything below this
# floor to the floor value so they're at least visible. 0.125 = a 32nd note,
# which is the smallest duration users typically see in falling-notes UIs.
_MIN_VISIBLE_DURATION = 0.125


def musicxml_to_payload(
    xml_path: Path,
    fallback_time_signature: str | None = None,
) -> dict[str, Any]:
    """Convert an Audiveris MusicXML file to the NoteEvent payload.

    `fallback_time_signature` (e.g. "4/4", "2/4") gets injected at
    offset 0 of every part when the score has no detected time
    signature of its own. This is the inherit-from-previous-page hook:
    main.py threads the time signature it learned from the first page
    forward to later pages whose MusicXML lacks one, so per-measure
    drift compensation actually fires on those pages instead of
    skipping silently.
    """
    score = music21.converter.parse(str(xml_path))

    # Inject the inherited time signature into each part if Audiveris
    # didn't emit one on this page. Place it at offset 0 of each part
    # so it governs all subsequent measures.
    if fallback_time_signature:
        existing = list(score.flatten().getElementsByClass(m21_meter.TimeSignature))
        if not existing:
            try:
                injected_count = 0
                for part in score.parts:
                    ts = m21_meter.TimeSignature(fallback_time_signature)
                    part.insert(0.0, ts)
                    injected_count += 1
                log.info("injected fallback time signature %s into %d part(s)",
                         fallback_time_signature, injected_count)
            except Exception as exc:  # noqa: BLE001
                log.warning("could not inject fallback time signature %s: %s",
                            fallback_time_signature, exc)

    # expandRepeats() unrolls forward/backward repeats, voltas, D.S./D.C./
    # Coda/Segno jumps into a linear timeline. Without this a piece with
    # a 16-bar verse repeat plays only the first 16 bars and ends. Raises
    # on malformed repeat structures; fall back to the un-expanded score
    # so a slightly broken file still plays.
    try:
        score = score.expandRepeats()
    except Exception:  # noqa: BLE001
        log.info("expandRepeats failed; using un-expanded score")

    # makeAccidentals applies the key signature and measure-scoped
    # accidental rules so a note Audiveris transcribed without an
    # explicit accidental still sounds at the right pitch when an earlier
    # note in the same measure had one. Also normalises chord notes so
    # adjacent stacked noteheads don't end up with inconsistent spellings.
    try:
        score.makeAccidentals(inPlace=True, overrideStatus=True)
    except Exception:  # noqa: BLE001
        log.info("makeAccidentals failed; pitches may inherit Audiveris spellings")

    # stripTies merges ties into single notes whose duration equals the
    # sum of the tied components. For playback that's the correct shape —
    # a tie is a notation device, not a re-articulation. Without this,
    # the visualiser shows the second half of every tied note as a new
    # attack, which sounds (and looks) wrong.
    try:
        score = score.stripTies(retainContainers=True)
    except Exception:  # noqa: BLE001
        log.info("stripTies failed; tied notes may emit as separate events")

    # Cumulative drift compensation. Audiveris occasionally emits a
    # measure whose notes sum to more than the time signature's bar
    # duration (a "broken measure"). Without intervention each such
    # excess accumulates forward, pushing every later measure offset by
    # the cumulative overflow. Ground-truth A/B against the MIDI of
    # Mozart's Turkish March showed pitch precision at 88% but
    # pitch+beat F1 at only 12% — the pitches were right but assigned
    # to drifted beats.
    #
    # This pass walks each part's measures in order, tracks cumulative
    # drift, shifts each measure's offset back by the drift carried
    # from prior broken measures, and clips notes within a broken
    # measure to stay within the bar. Measures with no drift before
    # them and no broken predecessor are untouched — much more
    # surgical than the per-measure rigid grid I tried first.
    drift_fixes = _compensate_cumulative_drift(score)
    if drift_fixes:
        log.info("drift-compensated %d measure(s)", drift_fixes)

    bpm = _detect_bpm(score)

    # Parts give us staff structure — for a piano grand staff Audiveris
    # emits two parts (top staff = right hand, bottom = left). For
    # single-staff scores with two voices (a common piano encoding for
    # right + left hand on one staff), voice id maps to hand. For all
    # other shapes we fall back to a middle-C pitch heuristic.
    parts = list(score.parts) if score.parts else [score]
    use_part_hand = len(parts) == 2

    total_dynamic_markings = 0
    events: list[dict[str, Any]] = []
    for part_idx, part in enumerate(parts):
        hand_for_part = None
        if use_part_hand:
            hand_for_part = "right" if part_idx == 0 else "left"

        # When the part isn't already split per-hand by parts, check for
        # multi-voice structure within measures. Two voices → upper voice
        # is right hand, lower is left.
        voice_hand_map = (
            {} if use_part_hand else _voice_hand_map(part)
        )

        flat = part.flatten()

        # Walk the part once to collect (offset, velocity) for every
        # dynamic marking. We then look up the active velocity for each
        # note by linear scan on this list.
        dyn_events: list[tuple[float, int]] = []
        for d in flat.getElementsByClass(m21_dynamics.Dynamic):
            vel = _dynamic_to_velocity(d.value)
            if vel is not None:
                dyn_events.append((float(d.offset), vel))
        dyn_events.sort(key=lambda x: x[0])
        total_dynamic_markings += len(dyn_events)

        for n in flat.notes:
            offset = float(n.offset)
            duration = float(n.duration.quarterLength)
            if duration <= 0:
                # Music21 sometimes gives grace notes zero duration —
                # they have musical meaning but no audible duration. Lift
                # them to the visible-duration floor instead of skipping;
                # otherwise the visualizer loses ornaments entirely.
                duration = _MIN_VISIBLE_DURATION
            elif duration < _MIN_VISIBLE_DURATION:
                # Tiny grace-note durations (e.g., 0.0625 = 64th note)
                # render as imperceptible flashes. Promote to a 32nd-note
                # floor so they're at least legible.
                duration = _MIN_VISIBLE_DURATION

            # Per-note hand: voice-derived if available, then part-derived,
            # then middle-C heuristic (handled inside _event).
            hand_override = hand_for_part
            if hand_override is None and voice_hand_map:
                hand_override = _hand_from_voice(n, voice_hand_map)

            base_vel = _velocity_at(dyn_events, offset, _DEFAULT_VELOCITY)
            vel_mul, dur_mul = _articulation_effects(n)
            velocity = max(1, min(127, int(round(base_vel * vel_mul))))
            effective_duration = duration * dur_mul

            if n.isChord:
                for p in n.pitches:
                    events.append(_event(
                        p, offset, effective_duration,
                        velocity=velocity, hand_override=hand_override,
                    ))
            else:
                events.append(_event(
                    n.pitch, offset, effective_duration,
                    velocity=velocity, hand_override=hand_override,
                ))

    # Audiveris occasionally emits the same pitch twice at the same
    # offset within a part — once as the "correct" duration and once
    # as a spurious shorter event from a voice or glyph parsing
    # artifact. The duplicates inflate measure durations past the time
    # signature and visibly clutter the falling-notes view (the user
    # sees a tall block AND a stubby block on the same key). Dedup by
    # (beat, note, hand), keeping the longest-duration entry as the
    # canonical one. Confirmed safe on clean inputs (Brahms test corpus
    # has zero duplicates → zero events removed).
    events, dedup_count = _dedupe_same_pitch_offset(events)
    if dedup_count:
        log.info("dedup: removed %d duplicate event(s)", dedup_count)

    # Triplet recovery. Audiveris frequently misreads triplet groups
    # — three events that should span 1 beat (durations ≈ 1/3) get
    # transcribed as three eighths (durations 0.5, sum 1.5) when the
    # tuplet bracket OCR fails. Gated on broken-measure context: only
    # fires when retiming would actually bring the containing measure
    # back to the time-signature bar duration, so legitimate 3-eighth
    # runs in 6/8 / 3/4 are left alone.
    bar_dur_for_triplets: float | None = None
    if parts:
        ts_list = list(parts[0].flatten().getElementsByClass(m21_meter.TimeSignature))
        if ts_list:
            bar_dur_for_triplets = float(ts_list[0].barDuration.quarterLength)
    triplet_fixes = _recover_triplets(events, bar_dur_for_triplets)
    if triplet_fixes:
        log.info("triplet recovery: re-timed %d group(s)", triplet_fixes)

    # Beat quantization. After drift compensation, most events are
    # already very close to standard subdivisions (sixteenths, eighths,
    # triplet boundaries). A conservative snap-to-grid pass cleans up
    # the residual float noise without touching legitimately off-grid
    # notes. Tolerance of 0.04 beats (1/25 of a quarter) is tight
    # enough that genuine rubato / unusual subdivisions stay
    # untouched, but tight enough to absorb the tiny drift residues
    # left after the per-measure compensation pass.
    snapped = _quantize_beats_to_grid(events)
    if snapped:
        log.info("quantize: snapped %d event(s) to grid", snapped)

    events.sort(key=lambda e: (e["beat"], e["note"]))
    # total_beats lets the caller stitch this page onto a multi-page
    # score without overlapping the next page's downbeat with this
    # page's tail.
    total_beats = float(score.duration.quarterLength or 0.0)
    meta = _score_meta(score, parts, total_dynamic_markings)
    meta["duplicates_removed"] = dedup_count
    log.info(
        "convert: %d events, bpm=%.1f, key=%s, time=%s, parts=%d, "
        "voice_hands=%s, dyn_markings=%d",
        len(events), bpm, meta.get("key"), meta.get("time_signature"),
        len(parts),
        bool(not use_part_hand and any(
            _voice_hand_map(p) for p in parts
        )),
        total_dynamic_markings,
    )
    measures_map, system_count = _measure_system_map(parts)
    return {
        "bpm": bpm,
        "notes": events,
        "total_beats": total_beats,
        "meta": meta,
        # Score-view sync data: chronological measure start-beats with
        # the printed system (line of music) each belongs to. Repeats are
        # already expanded, so a jump back to an earlier line falls out
        # naturally (the same printed measure appears twice with two
        # different beats).
        "measures": measures_map,
        "system_count": system_count,
    }


def _measure_system_map(parts) -> tuple[list[dict[str, Any]], int]:
    """Chronological `{beat, system}` for each measure of the lead part,
    plus the number of printed systems (from MusicXML new-system layout
    breaks). Audiveris always emits system breaks for its own layout, so
    the count matches what's physically on the page; if none are found
    the page is a single system.

    Measure *numbers* index the printed page (stable across repeat
    expansion); measure *offsets* are playback beats. Mapping goes
    number → system via the sorted break-numbers list."""
    if not parts:
        return [], 1
    measures = list(parts[0].getElementsByClass(m21_stream.Measure))
    if not measures:
        return [], 1

    break_numbers: list[int] = []
    seen: set[int] = set()
    for m in measures:
        num = int(m.number or 0)
        if num in seen:
            continue  # repeat expansion revisits printed measures
        seen.add(num)
        for sl in m.getElementsByClass(m21_layout.SystemLayout):
            if getattr(sl, "isNew", False):
                break_numbers.append(num)
                break
    break_numbers.sort()
    # A new-system flag on the very first measure marks the layout start,
    # not a break — dropping it keeps system indices 0-based and the
    # count honest.
    first_num = int(measures[0].number or 0)
    if break_numbers and break_numbers[0] <= first_num:
        break_numbers.pop(0)

    out = []
    for m in measures:
        num = int(m.number or 0)
        out.append({
            "beat": float(m.offset),
            "system": bisect_right(break_numbers, num),
            # Printed measure identity — repeat expansion reuses numbers,
            # letting downstream tell "5 plays of this line" apart from
            # "5 printed measures on this line".
            "number": num,
        })
    out.sort(key=lambda e: e["beat"])
    return out, len(break_numbers) + 1


def _compensate_cumulative_drift(score) -> int:
    """Compensate for cumulative beat drift caused by broken (over-long)
    measures. For each part:

    - Walk measures in order, tracking cumulative excess from prior
      broken measures.
    - Whenever cumulative excess > 0.05 beats, shift the current
      measure's absolute offset back by that amount.
    - For each broken measure (its own duration > bar_dur), clip notes
      whose end extends past the bar to prevent the drift from
      compounding further.

    Returns the count of measures whose offset got shifted. Untouched
    if no time signature is detected (we'd have no `bar_dur` to compare
    against). Anacrusis measures (first measure shorter than bar_dur by
    more than 1 beat) are left alone — they're legitimate pickups, not
    rhythm errors.
    """
    shifted = 0
    for part in score.parts:
        ts_list = list(part.flatten().getElementsByClass(m21_meter.TimeSignature))
        if not ts_list:
            continue
        bar_dur = float(ts_list[0].barDuration.quarterLength)
        if bar_dur <= 0:
            continue

        measures = list(part.getElementsByClass(m21_stream.Measure))
        if not measures:
            continue

        cum_drift = 0.0
        first_dur = float(measures[0].duration.quarterLength)
        is_anacrusis = first_dur < bar_dur - 1.0

        for idx, m in enumerate(measures):
            # Skip the pickup measure entirely — it's legitimately
            # short and we don't want to mistake it for drift.
            if idx == 0 and is_anacrusis:
                continue

            # If prior measures have accumulated drift, shift this
            # measure's absolute offset back by that amount. Internal
            # note offsets (relative to the measure) stay the same.
            if cum_drift > 0.05:
                try:
                    m.offset = float(m.offset) - cum_drift
                    shifted += 1
                except Exception:  # noqa: BLE001
                    pass

            # Detect this measure's own excess and clip overflowing
            # notes so the excess doesn't carry forward beyond what
            # this measure absorbed.
            this_dur = float(m.duration.quarterLength)
            this_excess = this_dur - bar_dur
            if this_excess > 0.05:
                for n in m.flatten().notes:
                    local_offset = float(n.offset)
                    local_end = local_offset + float(n.duration.quarterLength)
                    if local_end > bar_dur + 0.05:
                        new_dur = max(_MIN_VISIBLE_DURATION,
                                      bar_dur - local_offset)
                        try:
                            n.duration.quarterLength = new_dur
                        except Exception:  # noqa: BLE001
                            pass
                # The excess pushed everything after this measure; add
                # it to the cumulative drift carried forward.
                cum_drift += this_excess
    return shifted


def _recover_triplets(
    events: list[dict[str, Any]],
    bar_dur: float | None = None,
) -> int:
    """Detect groups of 3 consecutive same-duration eighth notes that
    should actually be triplet eighths, and re-time them.

    The trigger: 3 same-duration eighths spanning 1.5 beats starting on
    a beat boundary. Audiveris's most common triplet misread is to lose
    the tuplet bracket and emit the group as regular eighths (each
    duration 0.5 instead of 1/3), so the group spans 1.5 beats instead
    of the 1.0 beat a true triplet occupies.

    Safeguard against false positives on legitimate 3-eighth runs (e.g.
    in 6/8 or 3/4 where 3 eighths spanning 1.5 beats is normal): only
    fire when retiming the group would FIX an over-long measure — i.e.,
    only when the measure containing the group has total content
    exceeding `bar_dur`, and shortening this group by 0.5 brings it
    closer to the expected bar duration.

    When `bar_dur` is None (no time signature detected), we conservatively
    skip — without measure context we can't tell triplets from legitimate
    runs.

    Returns the count of triplet groups retimed.
    """
    if len(events) < 3 or not bar_dur or bar_dur <= 0:
        return 0

    # Build per-measure totals across all events (any hand). Used to
    # decide whether retiming a candidate group would actually help.
    measure_durs: dict[int, float] = {}
    for e in events:
        m_idx = int(float(e["beat"]) // bar_dur)
        end = float(e["beat"]) + float(e["dur"])
        prev_end = measure_durs.get(m_idx, 0.0)
        # Track the *latest end* within the measure relative to its
        # start, which mirrors music21's measure-duration computation.
        local_end = end - m_idx * bar_dur
        if local_end > prev_end:
            measure_durs[m_idx] = local_end

    by_hand: dict[str, list[dict[str, Any]]] = {}
    for e in events:
        by_hand.setdefault(e.get("hand") or "", []).append(e)

    eps = 0.02
    fixes = 0
    for hand_events in by_hand.values():
        hand_events.sort(key=lambda e: e["beat"])
        i = 0
        while i <= len(hand_events) - 3:
            a, b, c = hand_events[i], hand_events[i + 1], hand_events[i + 2]
            if (abs(a["dur"] - 0.5) < eps
                and abs(b["dur"] - 0.5) < eps
                and abs(c["dur"] - 0.5) < eps
                and abs(b["beat"] - (a["beat"] + 0.5)) < eps
                and abs(c["beat"] - (b["beat"] + 0.5)) < eps
                and abs(a["beat"] - round(a["beat"])) < eps):
                start = a["beat"]
                m_idx = int(start // bar_dur)
                m_dur = measure_durs.get(m_idx, 0.0)
                # Only fire when the containing measure is broken
                # (over-long) AND retiming this group by -0.5 would
                # bring the measure closer to the expected bar duration.
                if m_dur > bar_dur + 0.3 and m_dur - 0.5 >= bar_dur - 0.1:
                    a["dur"] = b["dur"] = c["dur"] = 1.0 / 3.0
                    b["beat"] = start + 1.0 / 3.0
                    c["beat"] = start + 2.0 / 3.0
                    fixes += 1
                    measure_durs[m_idx] -= 0.5
                    i += 3
                    continue
            i += 1
    return fixes


def _quantize_beats_to_grid(
    events: list[dict[str, Any]], max_snap: float = 0.04
) -> int:
    """Snap each event's beat to the nearest standard subdivision when
    within `max_snap` of one. Standard subdivisions per beat: 16ths,
    8ths, dotted 8ths, triplet 8ths, quarters, etc.

    Conservative by design — `max_snap=0.04` (≈1/25 of a quarter)
    leaves anything more than a few percent off the grid alone,
    preserving legitimate rubato and unusual subdivisions. Returns the
    count of events that were snapped.

    Operates on the per-beat fractional part so it works at any
    absolute beat position; also considers the next integer beat
    (frac → 1.0) to handle near-end-of-beat positions.
    """
    # Per-beat grid lines: 0, 16ths, 12ths (triplets), 8ths, dotted-8th,
    # quarters, half, plus their complements. Listing dense to catch
    # any reasonable subdivision musicians actually use.
    grid = sorted({
        0.0,
        1/16, 1/12, 1/8, 1/6, 3/16,
        1/4, 1/3,
        5/16, 3/8, 5/12, 7/16,
        1/2,
        9/16, 7/12, 5/8, 2/3, 11/16,
        3/4, 13/16, 5/6, 7/8, 11/12, 15/16,
    })
    snapped = 0
    for e in events:
        beat = float(e["beat"])
        whole = int(beat) if beat >= 0 else -(-int(-beat))
        frac = beat - whole
        # Find closest grid point including wrap to next whole beat.
        best = frac
        best_err = float("inf")
        for g in grid:
            err = abs(frac - g)
            if err < best_err:
                best_err = err
                best = g
        # Next-beat boundary (1.0)
        if abs(frac - 1.0) < best_err:
            best_err = abs(frac - 1.0)
            best = 1.0
        if 1e-9 < best_err <= max_snap:
            e["beat"] = float(whole + 1) if best == 1.0 else float(whole) + best
            snapped += 1
    return snapped


def _dedupe_same_pitch_offset(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Collapse events that share `(beat, note, hand)` to a single
    canonical entry — the one with the longest duration. Returns
    `(deduped_events, count_removed)`.

    Used to clean up Audiveris's occasional duplicate emissions where
    the same pitch is transcribed twice at the same offset (e.g., a
    misread overlapping glyph, a multi-voice artifact). On clean
    transcriptions this is a no-op (no duplicates to remove); on noisy
    ones it noticeably tightens broken measures by removing the short
    stragglers that pushed the measure duration past the time signature.

    Note: keying on `(beat, note, hand)` deliberately preserves notes
    at the same offset in *different* hands — those are legitimate
    chord-spread-across-hands cases that the visualizer renders
    correctly.
    """
    canonical: dict[tuple, dict[str, Any]] = {}
    eps = 1e-4
    for e in events:
        key = (round(float(e["beat"]) / eps) * eps,
               e.get("note"),
               e.get("hand"))
        existing = canonical.get(key)
        if existing is None or float(e["dur"]) > float(existing["dur"]):
            canonical[key] = e
    removed = len(events) - len(canonical)
    return list(canonical.values()), removed


def _voice_hand_map(part) -> dict[str, str]:
    """Returns a `{voice_id: hand}` mapping when the part has exactly two
    voices that can be assigned to right + left hands. Sorts voice ids
    so the *lexically smallest* (usually '1') goes to the right hand,
    which matches MusicXML convention (top stem = upper voice = voice 1)."""
    voice_ids: set[str] = set()
    for measure in part.getElementsByClass(m21_stream.Measure):
        for voice in measure.voices:
            if voice.id is not None:
                voice_ids.add(str(voice.id))
    if len(voice_ids) != 2:
        return {}
    sorted_ids = sorted(voice_ids)
    return {sorted_ids[0]: "right", sorted_ids[1]: "left"}


def _hand_from_voice(note, voice_hand_map: dict[str, str]) -> str | None:
    """Look up the active Voice context for this note and resolve to a
    hand. Returns None when no Voice container is in scope (e.g., the
    note lives directly under the Measure, no voicing)."""
    try:
        voice = note.getContextByClass(m21_stream.Voice)
    except Exception:  # noqa: BLE001
        return None
    if voice is None or voice.id is None:
        return None
    return voice_hand_map.get(str(voice.id))


def _score_meta(score, parts, dynamic_count: int) -> dict[str, Any]:
    """Diagnostic snapshot: detected key, time sig, measure count, etc.
    Useful for tracking down "why was this scan wrong" without re-running
    the whole pipeline. Lives under the response's `meta` field — purely
    informational; the visualizer doesn't read it."""
    meta: dict[str, Any] = {
        "parts": len(parts),
        "dynamic_markings": dynamic_count,
    }
    try:
        flat = score.flatten()
        # Key — prefer the first KeySignature; fall back to estimated.
        ks_list = list(flat.getElementsByClass(m21_key.KeySignature))
        if ks_list:
            ks = ks_list[0]
            meta["key"] = (
                ks.asKey().tonicPitchNameWithCase
                if hasattr(ks, "asKey") else str(ks)
            )
        # Time signature — first occurrence.
        ts_list = list(flat.getElementsByClass(m21_meter.TimeSignature))
        ts = ts_list[0] if ts_list else None
        if ts is not None:
            meta["time_signature"] = ts.ratioString
        # Measure count + how many measures don't sum to the time
        # signature (a proxy for rhythm-parsing errors). Computed once
        # over the first part since all parts share measure boundaries.
        if parts:
            first_part = parts[0]
            measures = list(
                first_part.getElementsByClass(m21_stream.Measure)
            )
            if measures:
                meta["measures"] = len(measures)
                meta["broken_measures"] = _count_broken_measures(measures, ts)
    except Exception as exc:  # noqa: BLE001
        # Diagnostic only — never fail the scan over a missing meta field.
        log.debug("score meta extraction failed: %s", exc)
    return meta


def _count_broken_measures(measures, time_signature) -> int:
    """Count measures whose summed note/rest duration doesn't match the
    active time signature. A "broken" measure usually means Audiveris
    misclassified at least one duration in that measure (e.g., a beamed
    eighth read as a quarter), so this count is a useful proxy for
    rhythm-parsing accuracy on the page.

    Generous tolerance (0.05 beats ~ 1/20th of a quarter note) so we
    don't flag floating-point noise as broken. Returns 0 if no time
    signature was detected.
    """
    if time_signature is None:
        return 0
    expected = float(time_signature.barDuration.quarterLength)
    if expected <= 0:
        return 0
    broken = 0
    for m in measures:
        actual = float(m.duration.quarterLength)
        # Pickup (anacrusis) measures are legitimately shorter than the
        # time signature — don't flag those. Heuristic: only flag when
        # the actual is larger than expected, or smaller by more than
        # one beat (which is too much to be a pickup).
        if actual > expected + 0.05:
            broken += 1
        elif actual < expected - 1.0:
            broken += 1
    return broken


def _detect_bpm(score) -> float:
    """Walk the score for tempo information, preferring explicit metronome
    marks (♩=120) and falling back to text indications ("Allegro",
    "Andante") via music21's tempo-text → BPM mapping.

    Audiveris emits bold-italic words ("Allegro", "Andante", etc.) as
    plain `TextExpression` elements inside `<direction>` blocks, NOT as
    `<metronome>` or `<sound tempo="">`. music21 doesn't auto-promote
    those to `TempoText`, so we have to do that ourselves. Without this
    pass, scores like Brahms's Hungarian Dance — whose only tempo
    indication is the word "Allegro" — silently fall back to our default
    90 BPM.
    """
    flat = score.flatten()

    # Explicit metronome marks first — these carry an actual number.
    for mark in flat.getElementsByClass(m21_tempo.MetronomeMark):
        if mark.number:
            return float(mark.number)

    # Already-tagged tempo text (rare from Audiveris, common from
    # hand-edited MusicXML).
    for text in flat.getElementsByClass(m21_tempo.TempoText):
        bpm = _bpm_from_tempo_text(text.text)
        if bpm is not None:
            return bpm

    # The Audiveris case: plain TextExpression elements that happen to
    # contain a tempo word. Scan and promote.
    for expr in flat.getElementsByClass(m21_expressions.TextExpression):
        bpm = _bpm_from_tempo_text(expr.content)
        if bpm is not None:
            return bpm

    return _DEFAULT_BPM


def _bpm_from_tempo_text(text: str | None) -> float | None:
    """Map a raw text string ("Allegro", "Andante con moto") to a BPM via
    music21's tempo lookup. Returns None when the string doesn't match a
    known tempo term. Strips bold/italic styling artifacts and runs the
    lookup against the normalized text."""
    if not text:
        return None
    # Audiveris occasionally returns the text trimmed of an opening
    # consonant from a stylized capital (e.g., "llegro" instead of
    # "Allegro"). Try both the raw text and a few common reconstructions.
    candidates = {text.strip()}
    candidates.add(text.strip().lower())
    # Strip leading lowercase letters that look like a concatenated
    # dynamic mark (e.g., "fpassionato" — the leading 'f' is a forte,
    # the rest is "passionato" which isn't a tempo).
    s = text.strip().lower()
    if s and s[0] in ("f", "p", "m") and len(s) > 2:
        candidates.add(s[1:])
        if len(s) > 3 and s[1] in ("f", "p", "m"):
            candidates.add(s[2:])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            tt = m21_tempo.TempoText(candidate)
            mm = tt.getMetronomeMark()
        except Exception:  # noqa: BLE001
            continue
        if mm and mm.number:
            return float(mm.number)
    return None


def _dynamic_to_velocity(value: str | None) -> int | None:
    if not value:
        return None
    return _DYNAMIC_VELOCITY.get(value.strip().lower())


def _velocity_at(
    events: list[tuple[float, int]], offset: float, default: int
) -> int:
    """Find the most recent dynamic at or before `offset`. Linear scan is
    fine — most scores have well under 100 dynamic markings per part."""
    current = default
    for d_off, d_vel in events:
        if d_off <= offset:
            current = d_vel
        else:
            break
    return current


def _articulation_effects(note) -> tuple[float, float]:
    """Combined (velocity_multiplier, duration_multiplier) for a note's
    articulations and expressions. Multipliers apply on top of the
    dynamic-derived base velocity, so an accent inside an mf passage is
    louder than an accent inside a p passage."""
    vel_mul = 1.0
    dur_mul = 1.0
    for art in getattr(note, "articulations", []) or []:
        eff = _ARTICULATION_TABLE.get(type(art).__name__)
        if eff is None:
            continue
        vel_mul *= eff[0]
        dur_mul *= eff[1]
    # Fermata appears under note.expressions in many scores, not
    # articulations — extend duration the same way.
    for exp in getattr(note, "expressions", []) or []:
        if type(exp).__name__ == "Fermata":
            dur_mul *= 2.0
    return vel_mul, dur_mul


def _event(
    pitch,
    beat: float,
    dur: float,
    velocity: int,
    hand_override: str | None,
) -> dict[str, Any]:
    # When we know the staff (2-part piano grand staff) use that;
    # otherwise fall back to the middle-C split used elsewhere for
    # single-track imports. Crude for non-piano scores, but adequate
    # for the visualizer's hand-mute toggle.
    if hand_override is not None:
        hand = hand_override
    else:
        hand = "right" if pitch.midi >= 60 else "left"
    return {
        "note": _pitch_str(pitch),
        "beat": beat,
        "dur": dur,
        "hand": hand,
        "vel": max(1, min(127, velocity)),
    }


def _pitch_str(pitch) -> str:
    """Scientific notation, sharps preferred. Matches the format the
    Flutter client expects (see OmrService._noteToMidi)."""
    name = pitch.step
    alter = int(pitch.alter or 0)
    if alter == 1:
        name += "#"
    elif alter == -1:
        name += "b"
    elif alter == 2:
        name += "##"
    elif alter == -2:
        name += "bb"
    return f"{name}{pitch.octave}"
