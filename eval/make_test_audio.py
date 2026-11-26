"""Make synthetic test clips + exact reference MIDI for evaluation.

Melody: "Twinkle Twinkle Little Star" (Ah! vous dirai-je, maman, 1761, public domain).
Run from repo root:  python eval/make_test_audio.py
Writes to eval/data/: easy_violin(.wav/.mid), easy_violin_take2.wav, easy_piano(.wav/.mid),
hard_fast_phone.wav + hard_fast.mid

These are SYNTHESIZED sounds, label them as such on Devpost. Add at least one real
recording of yourself too.
"""
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

SR = 22050
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

# (midi pitch, beats). C C G G A A G- | F F E E D D C-
TWINKLE = [(60, 1), (60, 1), (67, 1), (67, 1), (69, 1), (69, 1), (67, 2),
           (65, 1), (65, 1), (64, 1), (64, 1), (62, 1), (62, 1), (60, 2),
           (67, 1), (67, 1), (65, 1), (65, 1), (64, 1), (64, 1), (62, 2)]
# violin plays it an octave up (G3 is the lowest violin note)
VIOLIN = [(p + 12, b) for p, b in TWINKLE]
# piano left hand: one chord root per bar
PIANO_LH = [(48, 4), (53, 2), (48, 2), (53, 2), (48, 2), (43, 2), (48, 2), (48, 2), (53, 2), (48, 2), (43, 2)]


def hz(p):
    return 440 * 2 ** ((p - 69) / 12)


def violin_tone(p, dur, rng):
    t = np.arange(int(dur * SR)) / SR
    vib = 1 + 0.004 * np.sin(2 * np.pi * 5.5 * t) * np.clip(t / 0.25, 0, 1)  # vibrato after attack
    phase = 2 * np.pi * hz(p) * np.cumsum(vib) / SR
    y = sum((0.6 / k) * np.sin(k * phase) for k in range(1, 9))  # bowed, many harmonics
    env = np.minimum(1, t / 0.04) * np.minimum(1, (dur - t) / 0.06)
    return y * env * rng.uniform(0.8, 1.0)


def piano_tone(p, dur, rng):
    t = np.arange(int(dur * SR)) / SR
    y = sum((0.7 / k**1.3) * np.sin(2 * np.pi * k * hz(p) * t) for k in range(1, 7))
    env = np.exp(-3 * t) * np.minimum(1, t / 0.005) * np.minimum(1, (dur - t) / 0.03)
    return y * env * rng.uniform(0.8, 1.0)


def render(lines, bpm, tone, jitter=0.0, seed=0, lead=0.5):
    """lines: list of note lists. Returns audio + list of (start, end, pitch) truth."""
    rng = np.random.default_rng(seed)
    spb = 60 / bpm
    total = lead + max(sum(b for _, b in line) for line in lines) * spb + 1
    audio = np.zeros(int(total * SR))
    truth = []
    for line in lines:
        t = lead
        for p, beats in line:
            start = max(0.0, t + rng.uniform(-jitter, jitter))
            dur = beats * spb * 0.92
            y = tone(p, dur, rng)
            i = int(start * SR)
            audio[i:i + len(y)] += y
            truth.append((start, start + dur, p))
            t += beats * spb
    return audio / (np.abs(audio).max() + 1e-9) * 0.8, truth


def write(name, audio, truth=None, program=40):
    sf.write(OUT / f"{name}.wav", audio, SR)
    if truth:
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=program)
        inst.notes = [pretty_midi.Note(100, p, s, e) for s, e, p in truth]
        pm.instruments.append(inst)
        pm.write(str(OUT / f"{name}.mid"))


# 1. easy violin, 90 BPM
a, tr = render([VIOLIN], 90, violin_tone, seed=1)
write("easy_violin", a, tr)

# 2. repeat take: same melody, small human timing wobble (+-20 ms, inside the 50 ms tolerance)
a, _ = render([VIOLIN], 90, violin_tone, jitter=0.02, seed=2)
write("easy_violin_take2", a)

# 3. easy piano: melody + left-hand chords, 90 BPM
a, tr = render([TWINKLE, PIANO_LH], 90, piano_tone, seed=3)
write("easy_piano", a, tr, program=0)

# 4. hard: Mozart-style fast variation (16th notes), 120 BPM, room noise + echo like a phone recording
fast = []
for p, b in VIOLIN[:7]:
    for _ in range(int(b * 4)):
        fast.append((p, 0.25))
    fast[-1] = (p + 2, 0.25)  # neighbour note to make it busier
a, tr = render([fast], 120, violin_tone, jitter=0.01, seed=4)
echo = np.zeros_like(a)
d = int(0.06 * SR)
echo[d:] = 0.35 * a[:-d]
noise = np.random.default_rng(5).normal(0, 0.03, len(a))
write("hard_fast_phone", np.clip(a + echo + noise, -1, 1))
write("hard_fast", np.zeros(SR), tr)  # MIDI truth only
(OUT / "hard_fast.wav").unlink()

print("Wrote:", sorted(x.name for x in OUT.iterdir()))
