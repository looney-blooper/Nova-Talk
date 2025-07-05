from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

def get_model_tokenizer(model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype = torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model,tokenizer