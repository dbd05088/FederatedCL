# environment setting
- Make sure your nvidia-smi and nvcc version all over cuda 11.6
```
conda create -n fcl python=3.10
conda activate fcl
pip install transformers
pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu118
pip install packaging
pip install flash-attn --no-build-isolation
pip install peft
pip install bitsandbytes
pip install pandas kornia opencv-python timm torch_optimizer easydict pycocoevalcap sentencepiece protobuf
```