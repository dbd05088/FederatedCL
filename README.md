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
pip install trl==0.8.6
pip install deepspeed=0.14.0
```

# How to run
**Many lines of code are commented/uncommented for variations. Make sure to check `cl_manager_client.py` and method python file before training or `eval_VLM.py` before evaluation.**

1. Train
- Client parallel running: `bash run_VLM.sh`
Arguments:
- Note: experiment name **All the models are saved in client_states_$NOTE folder**
- scenario

- Client for-loop running: `bash train_VLM.sh` --> **Recommand**
Arguments:
- Note: experiment name **All the models are saved in client_states_$NOTE folder**
- Mode: name of method (method in `federated_methods/method_manager.py`)
- NUM_ITER: number of updates per round in each client


2. Eval
`bash eval.sh`
Arguments:
- Note: experiment name **Models in client_states_$NOTE folder will be loaded**
- scenario
- round_to_eval: model weights ends with round{round_to_eval}.pth / round{round_to_eval-1}.pth for client/server will be loaded
