import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"; os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import snapshot_download
p=snapshot_download("Qwen/Qwen3-4B", local_dir=os.environ.get("MATS_ROOT", "/root/autodl-tmp/mats-r-lens") + "/assets/Qwen3-4B", allow_patterns=["*.json","*.safetensors","*.txt","merges.txt","vocab.json","tokenizer*"])
print("DLDONE", p)
