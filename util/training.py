from datasets import load_dataset
from transformers import LlamaTokenizer
from prompt_engineering import generate_prompt

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM, LlamaTokenizer
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model
from huggingface_hub import login

# getting model from HF
class Training(object):
    def __init__(self, model, data_src): 
        self.tokenizer = LlamaTokenizer.from_pretrained(model, add_eos_token=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # loading data from the local machine
        data = load_dataset("json", data_files=data_src)
        data = data.map(lambda data_point: {"prompt": self.tokenizer(generate_prompt(data_point))})

        model = LlamaForCausalLM.from_pretrained(
            model,
            load_in_8bit=True,
            device_map="auto",
        )

        model = prepare_model_for_int8_training(model)

        LORA_R = 4
        LORA_ALPHA = 16
        LORA_DROPOUT = 0.05
        config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(model, config)
        self.tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
        data = load_dataset("json", data_files=data_src)

        CUTOFF_LEN = 256  # 256 accounts for about 96% of the data
        self.data = data.shuffle().map(
            lambda data_point: self.tokenizer(
                generate_prompt(data_point),
                truncation=True,
                max_length=CUTOFF_LEN,
                padding="max_length",
            )
        )

    def training(self, output_dir, hf_token, dir_publish): 
        MICRO_BATCH_SIZE = 8  # change to 4 for 3090
        BATCH_SIZE = 128
        GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
        EPOCHS = 2  # paper uses 3
        LEARNING_RATE = 2e-5  # from the original paper
        

        trainer = transformers.Trainer(
            model=self.model,
            train_dataset=self.data["train"],
            args=transformers.TrainingArguments(
                per_device_train_batch_size=MICRO_BATCH_SIZE,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                warmup_steps=100,
                num_train_epochs=EPOCHS,
                learning_rate=LEARNING_RATE,
                fp16=True,
                logging_steps=1,
                output_dir=output_dir,
                save_total_limit=3,
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )
        self.model.config.use_cache = False
        trainer.train(resume_from_checkpoint=False)

        self.model.save_pretrained(output_dir)

        login(token=hf_token)
        self.model.push_to_hub(dir_publish, use_auth_token=True)
        print("model is successfully saved.")

        return self.model

if __name__=="__main__":
    
    print("=============")
    print('input model name [decapoda-research/llama-7b-hf]\n')
    model = input()
    print("model: " + model + '\n')
    print("=============")
    print('data source:\n')
    data_src = input()
    print("data source " + data_src + '\n')
    print("=============")
    print('output directory:\n')
    output_dir = input()
    print("output directory " + output_dir + '\n')
    print("=============")
    print('input model publication directory:\n')
    dir_publish = input()
    print("model publication directory " + dir_publish + '\n')
    print("=============")
    print('input Hugging Face token:\n')
    hf_token = input()
    print("=============")

    obj = Training(model, data_src)
    print(obj.training(output_dir, hf_token, dir_publish))


    
