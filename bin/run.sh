SL_DIR=$(mktemp -d);  ./env/bin/python -m moshi.server --ssl "$SSL_DIR" --tools-dir ./tools --intent-model google/gemma-3-1b-it --intent-model-dtype bfloat16 --intent-min-tokens 8 --device cuda
