# Pilot: artifact / sink census on pretrained audio encoders (CPU, 2026-09-30)

Existence check for the session-1 gap report (now in archive/). Not paper results (13 clips/model).

```
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers soundfile librosa datasets
sh runall.sh                                   # HuBERT/WavLM/Whisper/AST census -> results/*.json
python acoustics.py results/<model>.json       # energy percentile of outlier / sink frames
python whisper30.py openai/whisper-base        # 30-s full-speech, silence-gap, short+pad inputs
python whisperpos.py openai/whisper-base 4     # are outlier positions fixed across contents?
python whisperenergy.py openai/whisper-base 4  # AUROC: low frame energy -> outlier
```

`results/pilot_log.txt` holds the per-layer printouts quoted in the report.
