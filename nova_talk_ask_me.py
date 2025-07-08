from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

model_path = "models/check_point-1"

model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype = torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path)

s = input("Enter your Question releted to AstroPhysics (or) Space Science : ")

prompt = "Answer the following Question\n"+s+"\nAnswer:\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = tokenizer.decode(model.generate(inputs["input_ids"],max_new_tokens=200)[0],skip_special_tokens=True)

print(outputs)