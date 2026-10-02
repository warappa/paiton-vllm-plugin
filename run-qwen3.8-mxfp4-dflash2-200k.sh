export PAITON_TARGET_DIR="$PWD/model-cache/qwen38-nvfp4"
export PAITON_DRAFT_DIR="$PWD/model-cache/qwen38-dflash2"
export PAITON_CACHE_DIR="$PWD/runtime-cache/qwen38-rocm10-200k"
export ROCR_VISIBLE_DEVICES=1
export PAITON_NGRAM_CODRAFT=1 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
./models/Qwen3.8-MXFP4-DFlash2/run-rocm10.sh --context 200000 --profile chat