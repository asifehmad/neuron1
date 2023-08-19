import os
import torch
import deepspeed
import time

from transformers import AutoTokenizer, AutoModelForCausalLM
from deepspeed.runtime.utils import see_memory_usage

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
hf_token = "<your_token>"
max_tokens = 2000
prompt = "Write a detailed note on AI"

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    use_fast=True,
    token=hf_token,
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    torch_dtype=torch.float16,
    device_map="cpu",
    token=hf_token,
)
see_memory_usage("After load model", force=True)

ds_model = deepspeed.init_inference(
    model=model,
    mp_size=world_size,
    dtype=torch.float16,
    replace_method="auto",
    replace_with_kernel_inject=True,
    max_out_tokens=max_tokens,
)
see_memory_usage("After DS-inference init", force=True)

torch.cuda.synchronize()
start = time.time()
inputs = tokenizer.encode(f"<human>: {prompt} \n<bot>:", return_tensors="pt").to(
    f"cuda:{local_rank}"
)
outputs = ds_model.generate(inputs, max_new_tokens=max_tokens)
output_str = tokenizer.decode(outputs[0])
torch.cuda.synchronize()
end = time.time()
see_memory_usage("After forward", force=True)

print("output:", output_str)
print("Inference Time:", end - start)
