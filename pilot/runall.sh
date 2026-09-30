for spec in "frame facebook/hubert-large-ll60k" "frame microsoft/wavlm-base-plus" "frame microsoft/wavlm-large" "whisper openai/whisper-base" "whisper openai/whisper-small" "ast MIT/ast-finetuned-audioset-10-10-0.4593"; do
  set -- $spec
  echo "#### $2"
  timeout 2400 python3 pilot.py $1 $2 2>&1 | grep -v -i "warn\|Loading\|it/s"
done
