import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
import transformers
from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM, LlamaTokenizer
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model

from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

class Inference(object):
    def __init__(self, model, dir_publish, offload_folder):
        self.tokenizer = LlamaTokenizer.from_pretrained(model)

        model = LlamaForCausalLM.from_pretrained(
            model,
            load_in_8bit=True,
            device_map="auto",
            offload_folder=offload_folder
        )
        self.model = PeftModel.from_pretrained(model, dir_publish)

    def gen_output(self, prompt, max_token=128):
        outputs=""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].cuda()

        generation_config = GenerationConfig(
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.15,
        )
        print("Generating...")
        generation_output = self.model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_token,
        )
        for s in generation_output.sequences:
            outputs += self.tokenizer.decode(s)
            print(self.tokenizer.decode(s))

        return outputs

if __name__=='__main__':
    print("=============")
    print('input model name [decapoda-research/llama-7b-hf]\n')
    model = input()
    print("model: " + model + '\n')
    print("=============")
    
    print('input model publication directory\n')
    dir_publish = input()
    print("publication directory: " + dir_publish + '\n')
    print("=============")

    print('input offload folder\n')
    offload_folder = input()
    print("offload folder: " + offload_folder + '\n')
    print("=============")

    print("example prompt \n")
    print('''
    When I failover my application, can I preserve the IP addresses for my applications?
    ''')
    print("=============")
    print('input your prompt:\n')
    prompt = input()
    print("=============")

    obj = Inference(model, dir_publish, offload_folder)
    print(obj.gen_output(prompt))
