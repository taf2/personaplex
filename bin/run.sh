INTENT_MODEL="${INTENT_MODEL:-google/gemma-3-1b-it}"
STT_PRESET="${STT_PRESET:-qwen}"

case "$STT_PRESET" in
  voxtral)
    USER_STT_MODEL="mistralai/Voxtral-Mini-4B-Realtime-2602"
    USER_STT_LANGUAGE_DEFAULT="en"
    ;;
  qwen)
    USER_STT_MODEL="Qwen/Qwen3-ASR-0.6B"
    USER_STT_LANGUAGE_DEFAULT="English"
    ;;
  *)
    USER_STT_MODEL="$STT_PRESET"
    USER_STT_LANGUAGE_DEFAULT="en"
    ;;
esac

USER_STT_LANGUAGE="${USER_STT_LANGUAGE:-$USER_STT_LANGUAGE_DEFAULT}"

SSL_DIR=$(mktemp -d); ./env/bin/python -m moshi.server --ssl "$SSL_DIR" --static ./client/dist --tools-dir ./tools --intent-model "$INTENT_MODEL" --intent-model-dtype bfloat16 --intent-min-tokens 8 --user-stt-model "$USER_STT_MODEL" --user-stt-dtype bfloat16 --user-stt-language "$USER_STT_LANGUAGE" --user-stt-chunk-seconds 2.0 --user-stt-stride-seconds 1.0 --device cuda
