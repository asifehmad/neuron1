import transformers
import tensor_parallel as tp
import torch
import time


tokenizer = transformers.AutoTokenizer.from_pretrained("asifhugs/open_llama_7b")
model = transformers.AutoModelForCausalLM.from_pretrained("asifhugs/open_llama_7b", torch_dtype=torch.half)  # use opt-125m for testing

model = tp.tensor_parallel(model, ["cuda:0", "cuda:1"])  # <- each GPU has half the weights

#inputs = tokenizer("A cat sat", return_tensors="pt")["input_ids"].to("cuda:0")
#outputs = model.generate(inputs, max_length=256)
#print(tokenizer.decode(outputs[0])) # A cat sat on my lap for a few minutes ...
prompt = 'Write a detailed note on AI'

start = time.time()
inputs = tokenizer.encode(f"<human>: {prompt} \n<bot>:", return_tensors="pt").to(
            model.device
            )
outputs = model.generate(inputs, max_new_tokens=300)
output_str = tokenizer.decode(outputs[0])
#torch.cuda.synchronize()
end = time.time()

print("output:", output_str)
print("Inference Time:", end - start)
