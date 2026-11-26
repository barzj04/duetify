# Duetify

Record each part of a duet, get one aligned score you can play from.
Audio → AI note detection (Basic Pitch) → cleaned, playable sheet music (music21) → shown in browser (OpenSheetMusicDisplay).

## Setup
Needs Python 3.10 or 3.11 (Basic Pitch does not support every newer version yet).

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows  (mac/linux: source .venv/bin/activate)
pip install -r requirements.txt
uvicorn app:app --reload
```
Open http://127.0.0.1:8000. No API keys, no paid services. Internet is only needed to load the sheet viewer script.

## How it works
| Step | File | What it does |
|---|---|---|
| Audio check | `pipeline.check_audio` | rejects silent, too long (>90s), unreadable files; detects tempo |
| AI transcription | `pipeline.transcribe` | Basic Pitch turns audio into notes with a confidence value |
| Cleaning | `pipeline.clean_notes` | removes notes the instrument can't play, flags low-confidence notes red, keeps violin as one melody line |
| Level | `pipeline.quantize` | Normal = 16th-note grid, Simple = 8th-note grid (drops tiny notes) |
| Duet merge | `pipeline.make_score` | aligns both parts to beat 0 and same tempo, piano on grand staff |

## Evaluation
Put audio + matching MIDI in `eval/data/`, list them in `eval/cases.csv`, then:
```bash
python eval/evaluate.py eval/cases.csv
```
Outputs a note-F1 table and an onset-threshold sweep.

## Limitations
- One instrument per recording. It does not separate piano and violin from one mixed recording.
- Parts recorded separately are aligned by first note, so start both recordings on beat 1.
- 4/4 time only, no key signature detection yet.
- Upload your own playing or public-domain music only. No YouTube downloading.

---

# Build checklist (deadline Nov 29, 7:00am MYT)

## Day 1 (Thu, today) – get it running
1. Make the GitHub repo `duetify`, public. Copy these files in. First commit **today** (commit dates prove event work).
2. Do the setup above. Test with any short recording of yourself.
3. If Basic Pitch install fails: try Python 3.10 in a fresh venv. Note the fix in README.
4. Record test clips on your phone (WAV or M4A, 10–30s each):
   - `easy_violin.wav` – slow simple tune (e.g. Twinkle, public domain)
   - `easy_violin_take2.wav` – same tune again (repeat check)
   - `easy_piano.wav` – simple piano piece
   - `hard_fast_phone.wav` – fast passage, phone in noisy room
5. Make reference MIDI for each: write the notes in MuseScore (free) → export `.mid`. Same tempo you played.

## Day 2 (Fri) – make it yours + evidence
6. Run `eval/evaluate.py`. Save the tables.
7. Pick the onset threshold with the best F1 from the sweep, set it as default in `pipeline.run`. **That's your A2 "project-specific AI decision" with evidence.**
8. Test the failure cases and screenshot each:
   - silent file, too-long file, `.txt` file, note out of violin range, very noisy clip
9. Test duet: violin clip + piano clip → one score. Screenshot.
10. Test Simple vs Normal on the same clip. Screenshot both (design/UX evidence: less clutter for beginners).
11. Style the page a bit (your own colours/logo text). Keep it simple.
12. Things to rewrite in your own words: page text, README intro, comments you want to change.

## Day 3 (Sat) – submit
13. Record the 2–4 min demo (screen record, OBS or Win+G):
   - 0:00–0:20 who it's for: you + friend duet, sheet music cost/time
   - 0:20–2:30 upload violin + piano → score appears → red notes → Simple toggle → download
   - 2:30–3:00 AI role: Basic Pitch + your cleaning; show F1 table
   - 3:00–3:30 built this week + limitations
   - Label any sped-up waiting part.
14. Upload to YouTube (unlisted is fine but embedding on), check in incognito.
15. Devpost page: answer the 5 questions (template below), add screenshots, team roster.
16. Fill the Participant Contact form.
17. Submit before **Sat Nov 28, 11pm MYT** to leave buffer.

## Devpost answer skeleton (rewrite in your words)
**1. Who is it for?** Amateur duet players and music students. My friend and I duet (piano + violin) and finding sheets for both parts is hard or paid. Transcribing by ear takes hours.
**2. What did we build?** New project, all built Nov 21–28. Pipeline, duet merge, range check, confidence flags, Simple level, web page, eval script. Third-party: Basic Pitch, music21, OSMD, librosa.
**3. AI use.** Finished project: Basic Pitch (lightweight, open source, runs on CPU, no API cost, handles any instrument). My decision: onset threshold chosen from sweep (table). Quality: note F1 table, 4 cases. Development: [name tool], what it wrote, what you checked/changed.
**4. Access + results.** Setup steps, cost free, results table (case / expected / actual / screenshot).
**5. Limits + failures.** Limitations list above + failure screenshots table.
