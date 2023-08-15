# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import bittensor
from typing import List, Dict
from transformers import LlamaForCausalLM, LlamaTokenizer
import deepspeed
import os

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '2'))

class OpenLlamaMiner(bittensor.HuggingFaceMiner):
    arg_prefix = "open_llama"
    system_label = "\nSystem:"
    assistant_label = "\nAssistant:"
    user_label = "\nUser:"

    def load_tokenizer(self):
        return LlamaTokenizer.from_pretrained(
            self.config.open_llama.model_name, use_fast=False
        )

    def load_model(self):
        model = LlamaForCausalLM.from_pretrained(
            self.config.open_llama.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True, device_map=local_rank
        )
        print('Loading Model for DeepSpeed now!')
        model = deepspeed.init_inference( model=model, mp_size=world_size, dtype=torch.float16, replace_method="auto", replace_with_kernel_inject=True, max_out_tokens=12000)
        print('Loaded the Model for DeepSpeed!')
        return model

    def forward(self, messages: List[Dict[str, str]]) -> str:
        history = self.process_history(messages)
        prompt = history + self.assistant_label

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        output = self.model.generate(
            input_ids,
            max_length=input_ids.shape[1] + self.config.open_llama.max_new_tokens)

        generation = self.tokenizer.decode(
            output[0][input_ids.shape[1] :], skip_special_tokens=True
        )
        generation = generation.split("User:")[0].strip()

        # Logging input and generation if debugging is active
        bittensor.logging.debug("Prompt: " + str(prompt))
        bittensor.logging.debug("Message: " + str(messages))
        bittensor.logging.debug("Generation: " + str(generation).replace("<", "-"))
        return generation


if __name__ == "__main__":
    bittensor.utils.version_checking()
    OpenLlamaMiner().run()
