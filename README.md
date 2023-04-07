# personalGPT

This tool intends to enpower delopement of personalized chatGPT like interface. The training data needs to be in json format with instruction-following demonstrations generated in the style of self-instruct using text-davinci-003.

We intend to offer the tool in different interfaces.

# cli Interface

## Training 

On running `python util/training.py`, a user is asked to input following details:

```
   * model [decapoda-research/llama-7b-hf]
   * data_src_directory
   * model_output_directory
   * model_publication_directory
   * token for Hugging Face (if the model is intended to be published in hf)
```

* currently, the tool has been tested only for "decapoda-research/llama-7b-hf" model.
* training data in `data_src_directory` must adhere to self-instruct style: [ref](https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json), [arxiv](https://arxiv.org/pdf/2212.10560.pdf)

## Inference 

Once the model is trained, a user can publish the model on a remote directory such as Hugging Face Hub or store it locally. 

On running `python inference.py`, a user is asked to input following details:

```
    * model[decapoda-research/llama-7b-hf]
    * model_publication_directory [local/remote]
    * offload_folder
    * input prompt
```
* here is a handy reference material for running large infernece on a consumer-grade gpu machine [ref](https://huggingface.co/docs/accelerate/usage_guides/big_modeling).

# Infrastructure

This tool requires NVIDIA CUDA environment.
