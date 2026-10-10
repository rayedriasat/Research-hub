# experiments/

All code is inference-only and runs on an RTX-3050 (4 GB) for Whisper ≤ large-v3-turbo; larger models / full benchmarks run on Kaggle T4.

| Folder | What | Status |
|---|---|---|
| `census_2026-09/` | Session-1 CPU census of registers / sinks in 7 audio encoders (13 inputs each). Indicative only. | done (v0) |
| `pilot_padding/` | Session-3 GPU pilots: what Whisper does with its 30-s padding; SilenceCache. | done (pilots v1–v3) |

## Environment
```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128   # Windows/Linux CUDA 12.8
.venv/Scripts/python -m pip install transformers accelerate soundfile librosa jiwer numpy scipy matplotlib pandas tqdm
```
Tested: torch 2.11.0+cu128, transformers 5.19.0, Python 3.14, RTX-3050 Laptop 4 GB.

## Data (git-ignored `data/`)
```bash
# LibriSpeech test-clean (346 MB)
curl -L -o data/test-clean.tar.gz https://www.openslr.org/resources/12/test-clean.tar.gz && tar -xzf data/test-clean.tar.gz -C data
# ESC-50 (non-speech, ~600 MB)
curl -L -o data/esc50.zip https://github.com/karoldvl/ESC-50/archive/master.zip && unzip data/esc50.zip -d data
```

## Re-running the pilots
```bash
cd experiments/pilot_padding
PYTHONPATH=. ../../.venv/Scripts/python whisper_regs.py   --model openai/whisper-base --n 100 --data ../../data/LibriSpeech/test-clean --out results
PYTHONPATH=. ../../.venv/Scripts/python whisper_padkv.py  --model openai/whisper-small --n 100 --data ../../data/LibriSpeech/test-clean --out results
PYTHONPATH=. ../../.venv/Scripts/python whisper_padkv.py  --model openai/whisper-large-v3-turbo --fp16 --n 100 --data ../../data/LibriSpeech/test-clean --out results
PYTHONPATH=. ../../.venv/Scripts/python whisper_nonspeech.py --model openai/whisper-base --n 200
```
Results (JSON, incl. 15 example transcripts per condition) are written to `pilot_padding/results/`; the numbers are summarised in `plan.md` §1.3 and `docs/evidence.md` (E22, E23).
