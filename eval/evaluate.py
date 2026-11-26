"""Measure transcription accuracy against a reference MIDI (ground truth).

Usage (from repo root):
  python eval/evaluate.py cases.csv
cases.csv columns: name,audio,reference_midi,instrument
  e.g. easy_violin,eval/data/twinkle.wav,eval/data/twinkle.mid,violin

Prints a markdown table (paste into Devpost) and sweeps the confidence threshold
so you can justify the value you chose (rubric A2 / A3).
"""
import csv
import sys
from pathlib import Path

import mir_eval
import numpy as np
import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import pipeline  # noqa: E402


def ref_notes(midi_path):
    pm = pretty_midi.PrettyMIDI(midi_path)
    ns = [n for inst in pm.instruments for n in inst.notes]
    iv = np.array([[n.start, n.end] for n in ns])
    hz = np.array([pretty_midi.note_number_to_hz(n.pitch) for n in ns])
    return iv, hz


def est_arrays(notes):
    iv = np.array([[n.start, n.end] for n in notes]) if notes else np.zeros((0, 2))
    hz = np.array([pretty_midi.note_number_to_hz(n.pitch) for n in notes])
    return iv, hz


def score(ref, est):
    # onset-only note F1 (50 ms tolerance), the standard MIR measure
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref[0], ref[1], est[0], est[1], offset_ratio=None
    )
    return p, r, f


def cleaned(raw, inst):
    notes = [pipeline.Note(n.start, n.end, n.pitch, n.conf) for n in raw]
    return pipeline.clean_notes(notes, inst)[0]


def main(cases_csv):
    rows = list(csv.DictReader(open(cases_csv)))
    print("| Case | Instrument | Ref notes | Raw AI F1 | After Duetify cleaning: Precision | Recall | Note F1 |")
    print("|---|---|---|---|---|---|---|")
    sweep = {t: [] for t in (0.3, 0.4, 0.5, 0.6, 0.7)}
    for row in rows:
        ref = ref_notes(row["reference_midi"])
        raw = pipeline.transcribe(row["audio"])
        f_raw = score(ref, est_arrays(raw))[2]
        p, r, f = score(ref, est_arrays(cleaned(raw, row["instrument"])))
        print(f"| {row['name']} | {row['instrument']} | {len(ref[1])} | {f_raw:.2f} | {p:.2f} | {r:.2f} | {f:.2f} |")
        for t in sweep:
            sweep[t].append(score(ref, est_arrays(cleaned(pipeline.transcribe(row["audio"], onset_threshold=t), row["instrument"])))[2])

    print("\nOnset threshold sweep, after cleaning (mean F1 across cases):")
    print("| Onset threshold | Mean F1 |\n|---|---|")
    for t, fs in sweep.items():
        print(f"| {t} | {np.mean(fs):.2f} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval/cases.csv")
