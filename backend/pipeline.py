"""Duetify pipeline: audio -> notes (Basic Pitch) -> cleaned score (music21) -> MusicXML.

Steps:
  1. check_audio      reject silent / too long / unreadable files
  2. transcribe       Basic Pitch gives note events (start, end, midi pitch, confidence)
  3. clean_notes      range check + low-confidence flag + level simplification
  4. build_part       quantize to beat grid and make a music21 part
  5. to_musicxml      single sheet or merged duet score
"""
from dataclasses import dataclass, field

import numpy as np
from music21 import chord, clef, instrument, layout, meter, note, stream, tempo

MAX_SECONDS = 90
SILENCE_RMS = 0.005

# playable MIDI range per instrument (piano A0-C8, violin G3 up to ~E7)
RANGES = {"piano": (21, 108), "violin": (55, 100)}

# level -> grid size in quarter notes, shortest kept note
LEVELS = {"normal": 0.25, "simple": 0.5}


@dataclass
class Note:
    start: float  # seconds
    end: float
    pitch: int  # MIDI number
    conf: float  # Basic Pitch amplitude, used as confidence (0-1)
    flags: list = field(default_factory=list)


# ---------- 1. audio checks ----------
def check_audio(path):
    """Return (bpm, warnings). Raises ValueError for files we cannot use."""
    import librosa

    try:
        y, sr = librosa.load(path, sr=22050, mono=True)
    except Exception as e:
        raise ValueError(f"Could not read audio file: {e}")
    secs = len(y) / sr
    if secs > MAX_SECONDS:
        raise ValueError(f"Clip is {secs:.0f}s. Please upload {MAX_SECONDS}s or less.")
    if secs < 1:
        raise ValueError("Clip is shorter than 1 second.")
    if float(np.sqrt(np.mean(y**2))) < SILENCE_RMS:
        raise ValueError("Recording seems silent. Check your mic or file.")
    bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(bpm)[0]) or 100.0
    return round(bpm), []


# ---------- 2. AI transcription ----------
def transcribe(path, onset_threshold=0.5):
    """Basic Pitch (Spotify, open source) -> list[Note]."""
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict

    _, _, events = predict(path, ICASSP_2022_MODEL_PATH, onset_threshold=onset_threshold)
    return [Note(float(s), float(e), int(p), float(a)) for s, e, p, a, *_ in events]


# ---------- 3. cleaning ----------
def clean_notes(notes, inst, conf_threshold=0.35):
    lo, hi = RANGES[inst]
    warnings = []
    out_of_range = 0
    for n in notes:
        if not lo <= n.pitch <= hi:
            n.flags.append("range")
            out_of_range += 1
        if n.conf < conf_threshold:
            n.flags.append("low_conf")
    # drop out-of-range notes, they cannot be played anyway
    kept = [n for n in notes if "range" not in n.flags]
    if out_of_range:
        warnings.append(f"Removed {out_of_range} note(s) outside the {inst} range.")
    kept, ghosts = remove_ghosts(kept)
    if ghosts:
        warnings.append(f"Removed {ghosts} overtone 'ghost' note(s).")
    low = sum("low_conf" in n.flags for n in kept)
    if low:
        warnings.append(f"{low} note(s) are low confidence and shown in red. Please check them.")
    if not kept:
        raise ValueError("No playable notes found. Try a clearer recording of one instrument.")
    if inst == "violin":
        kept = monophonic(kept)
    return kept, warnings


# intervals (semitones) where overtones of a real note land: octave, octave+5th, 2 octaves, 2 oct+3rd, 2 oct+5th
OVERTONES = (12, 19, 24, 28, 31)
GHOST_RATIO = 0.75  # a ghost is quieter than its parent by at least this factor
GHOST_MAX_CONF = 0.45  # and is itself weak (real notes in our tests were >= 0.5)


def remove_ghosts(notes, ratio=GHOST_RATIO, max_conf=GHOST_MAX_CONF):
    """Basic Pitch sometimes reports a note's overtones as extra notes.
    Drop a note if a clearly stronger note sits an overtone interval below it at the same time."""
    out, dropped = [], 0
    for n in notes:
        ghost = any(
            n.pitch - m.pitch in OVERTONES
            and abs(n.start - m.start) < 0.15
            and n.conf < max_conf
            and n.conf < m.conf * ratio
            for m in notes
        )
        if ghost:
            dropped += 1
        else:
            out.append(n)
    return out, dropped


def fit_bpm(notes, bpm, grid=0.25):
    """Fine-tune the detected tempo (+-4 BPM) so note starts land closest to the beat grid."""
    starts = np.array(sorted({round(n.start, 3) for n in notes}))
    starts = starts - starts[0]
    best, best_err = bpm, 1e9
    for cand in np.arange(bpm - 4, bpm + 4.01, 0.1):
        beats = starts * cand / 60 / grid
        err = np.mean(np.abs(beats - np.round(beats)))
        if err < best_err:
            best, best_err = cand, err
    return float(best)


def monophonic(notes):
    """Violin v1: keep the highest note when notes overlap (melody line)."""
    notes = sorted(notes, key=lambda n: (n.start, -n.pitch))
    out = []
    for n in notes:
        if out and n.start < out[-1].end - 0.05:
            if n.pitch > out[-1].pitch:
                out[-1].end = n.start  # cut previous, new note is higher
                out.append(n)
            continue
        out.append(n)
    return [n for n in out if n.end > n.start]


# ---------- 4. score building ----------
def quantize(notes, bpm, level, shift=0.0):
    """Seconds -> quarter-note offsets on a grid. Drops notes shorter than the grid."""
    grid = LEVELS[level]
    q = []
    for n in notes:
        off = round(((n.start - shift) * bpm / 60) / grid) * grid
        dur = round(((n.end - n.start) * bpm / 60) / grid) * grid
        if dur < grid:
            continue
        q.append((max(off, 0.0), dur, n))
    return q


def build_part(qnotes, inst, part_cls=stream.Part):
    part = part_cls()
    part.insert(0, instrument.Violin() if inst == "violin" else instrument.Piano())
    # group notes that start together into chords
    by_off = {}
    for off, dur, n in qnotes:
        by_off.setdefault(off, []).append((dur, n))
    for off in sorted(by_off):
        group = by_off[off]
        dur = min(d for d, _ in group)
        pitches = sorted({n.pitch for _, n in group})
        el = note.Note(spell(pitches[0])) if len(pitches) == 1 else chord.Chord([spell(p) for p in pitches])
        el.quarterLength = dur
        if any("low_conf" in n.flags for _, n in group):
            el.style.color = "#d62728"
        part.insert(off, el)
    # cut overlaps so each part is a single line of notes/chords
    els = list(part.notes)
    for a, b in zip(els, els[1:]):
        if a.offset + a.quarterLength > b.offset:
            a.quarterLength = max(b.offset - a.offset, LEVELS["normal"])
    return part


def spell(midi):
    """MIDI number -> pitch with no accidental for white keys (avoids natural signs everywhere)."""
    from music21 import pitch

    p = pitch.Pitch(midi=midi)
    if p.accidental is not None and p.accidental.name == "natural":
        p.accidental = None
    return p


def fill_rests(part, total):
    """Fill gaps with rests and pad to the full length so every staff lines up."""
    els = sorted(part.notes, key=lambda e: e.offset)
    t = 0.0
    for e in els:
        if e.offset > t:
            part.insert(t, note.Rest(quarterLength=e.offset - t))
        t = max(t, e.offset + e.quarterLength)
    if total > t:
        part.insert(t, note.Rest(quarterLength=total - t))
    return part


def piano_parts(qnotes):
    """Split into right hand (>= middle C) and left hand for a grand staff."""
    rh = build_part([q for q in qnotes if q[2].pitch >= 60], "piano", stream.PartStaff)
    lh = build_part([q for q in qnotes if q[2].pitch < 60], "piano", stream.PartStaff)
    rh.insert(0, clef.TrebleClef())
    lh.insert(0, clef.BassClef())
    return [rh, lh]


# ---------- 5. output ----------
def make_score(tracks, bpm, level, title="Duetify sheet"):
    """tracks: list of (inst, notes). Each track is aligned so its first note is beat 0."""
    score = stream.Score()
    score.insert(0, meter_title(title))
    built = []  # (parts, is_piano)
    for inst, notes in tracks:
        shift = min(n.start for n in notes)
        q = quantize(notes, bpm, level, shift)
        built.append((piano_parts(q), True) if inst == "piano" else ([build_part(q, inst)], False))
    # every staff gets the same length, rounded up to a full 4/4 bar
    end = max((e.offset + e.quarterLength for parts, _ in built for p in parts for e in p.notes), default=4)
    total = float(np.ceil(end / 4) * 4)
    for parts, is_piano in built:
        for p in parts:
            p.insert(0, meter.TimeSignature("4/4"))
            score.insert(0, fill_rests(p, total))
        if is_piano:
            score.insert(0, layout.StaffGroup(parts, symbol="brace", barTogether=True))
    score.parts[0].insert(0, tempo.MetronomeMark(number=round(bpm)))
    return score.makeNotation()


def meter_title(title):
    from music21 import metadata

    md = metadata.Metadata()
    md.title = title
    md.composer = "Transcribed with Duetify"
    return md


def to_musicxml(score):
    from music21.musicxml.m21ToXml import GeneralObjectExporter

    return GeneralObjectExporter(score).parse().decode("utf-8")


def run(files, bpm=None, level="normal", conf_threshold=0.35, onset_threshold=0.5):
    """files: list of (path, inst). Returns dict for the API."""
    warnings, tracks = [], []
    detected = None
    for path, inst in files:
        b, w = check_audio(path)
        detected = detected or b
        notes = transcribe(path, onset_threshold)
        notes, w2 = clean_notes(notes, inst, conf_threshold)
        warnings += [f"[{inst}] {x}" for x in w + w2]
        tracks.append((inst, notes))
    if not bpm:  # auto: refine the beat tracker's guess using the note starts
        bpm = fit_bpm([n for _, ns in tracks for n in ns], detected, LEVELS[level])
    score = make_score(tracks, bpm, level)
    return {
        "musicxml": to_musicxml(score),
        "bpm": round(bpm),
        "warnings": warnings,
        "note_count": sum(len(n) for _, n in tracks),
    }
