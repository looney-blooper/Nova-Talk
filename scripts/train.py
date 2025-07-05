from transformers import Trainer, BitsAndBytesConfig, HfArgumentParser, DataCollatorForSeq2Seq, TrainingArguments
from peft import LoraConfig, LoraModel, TaskType, get_peft_model
from trl import SFTTrainer
import torch
import os
import time
import csv

def train_model(original_model,tokenizer,tokenized_dataset,peft_model_path):
    lora_config = LoraConfig(
        r = 32,
        lora_alpha = 32,
        target_modules = ["q","v"],
        lora_dropout = 0.1,
        bias= "none",
        task_type= TaskType.SEQ_2_SEQ_LM,
    )
    peft_model = get_peft_model(
        original_model,
        lora_config
    )
    peft_training_args = TrainingArguments(
        output_dir= "./model/training_no_1",
        logging_dir= "./outputs/logs",
        logging_steps= 100,
        auto_find_batch_size= True,
        learning_rate= 1e-4,
        num_train_epochs= 250,
        max_steps= 25000,
        report_to= "none"
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=peft_model)

    peft_trainer = Trainer(
        model = peft_model,
        args = peft_training_args,
        train_dataset = tokenized_dataset,
        data_collator= data_collator
    )

    history = peft_trainer.train()

    peft_trainer.model.save_pretrained(peft_model_path)
    tokenizer.save_pretrained(peft_model_path)

    history_path = "./outputs/logging_history.csv"
    with open(history_path,"w",newline="") as f:
        if hasattr(history,"metrics"):
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])  
            for key, value in history.metrics.items():
                writer.writerow([key, value])

    return history_path