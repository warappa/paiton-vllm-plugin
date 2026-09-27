podman run --rm \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --ipc=host \
    localhost/paiton-qwen38-vllm030:0.1 \
    python3 -c 'import torch,vllm; print("torch =",torch.__version__); print("hip =",torch.version.hip); print("vllm =",vllm.__version__); print("gpu =",torch.cuda.get_device_name(0))'