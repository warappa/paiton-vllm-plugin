export PAITON_NEO_CACHE="$PWD/model-cache/paiton-qwen38-neo-cache"
#export PAITON_DRAFT_DIR="$PWD/model-cache/qwen38-neo-coder-dflash2"
#export PAITON_CACHE_DIR="$PWD/runtime-cache/paiton-qwen38-neo-cache"
export ROCR_VISIBLE_DEVICES=1
export HIP_VISIBLE_DEVICES=1
export PAITON_NGRAM_CODRAFT=1 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
./models/Qwen3.8-NEO-CODER-MAX/serve-docker.sh -e HIP_VISIBLE_DEVICES=1