# Research Hub — What do speech transformers do with silence?

Attention sinks, registers and the hidden work of padding in Whisper-family speech transformers (target: ICML 2027).

- **`plan.md`** — the single source of truth: pitch, hypotheses, experiments, progress tracker, decision log, lab notebook.
- `docs/evidence.md` — verified evidence log (full-text reads of 21+ papers, pilot results).
- `docs/scoop_check_2026-10-10.md` (+ `docs/scoop_logs/`) — novelty / scoop verification.
- `experiments/` — pilot code and results (see `experiments/README.md`).
- `paper/` — paper draft in the supervisor's REVTeX template (`pdflatex main && bibtex main && pdflatex main ×2`).
- `papers/manifest.txt` — arXiv IDs; re-download PDFs + TeX with `python scripts/download_arxiv_papers.py --manifest papers/manifest.txt`.
- `scripts/` — paper download / TeX flattening / cleanup tools.

Git-ignored: `Papers_pdf/`, `Papers_Tex_Source/`, `data/`, `archive/` (old plans + pre-overhaul git bundle), `.venv/`.
