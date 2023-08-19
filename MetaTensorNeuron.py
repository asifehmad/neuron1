#####deepspeed meta tensor#####
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from transformers import LlamaForCausalLM, LlamaTokenizer
import transformers
import os
import torch.distributed as dist
import io
import json
from pathlib import Path

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

deepspeed.init_distributed()
rank = dist.get_rank()
print(f"========Current randk:{rank} || lcoal rank:{local_rank} || world size:{world_size}==========")

model_path = 'asifhugs/open_llama_7b'

config = AutoConfig.from_pretrained(model_path)
# checkpoints_json = model_path+"checkpoints.json"
# if rank==0:
#   with io.open(checkpoints_json, "w", encoding="utf-8") as f:
#     file_list = [str(entry) for entry in Path(model_path).rglob("*.[bp][it][n]") if entry.is_file()]
#     data = {"type": "BLOOM", "checkpoints": file_list, "version": 1.0}
#     json.dump(data, f)
#     dist.barrier()

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
with deepspeed.OnDevice(dtype=torch.float16, device="meta"):
  model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)

model = model.eval()
ds_model = deepspeed.init_inference(
model,
mp_size=world_size,
dtype=torch.float16, replace_method="auto", replace_with_kernel_inject=True, max_out_tokens=12000
# base_dir=model_path,
# replace_with_kernel_inject=True,
# checkpoint=checkpoints_json,
)

prompt = "Where is Hawaii?"

encoding = tokenizer(prompt, return_tensors="pt")
generation_config = transformers.GenerationConfig(
temperature=0.0,
top_k=20,
repetition_penalty=1.2,
)

input_ids = encoding["input_ids"].to(model.device)
result = ds_model.generate(
input_ids=input_ids,
generation_config=generation_config,
return_dict_in_generate=False,
output_scores=False,
max_new_tokens=512,
)

output = tokenizer.decode(result[0][len(input_ids[0]):])
if rank == 0:
  print(output)
