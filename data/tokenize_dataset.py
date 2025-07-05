def tokenize_dataset(dataset, tokenizer):
    def tokenize_example(example):
        start_prompt = "Answer the following question:\n"
        end_prompt = "\nAnswer:\n"
        prompt = [start_prompt+q+end_prompt for q in example["question"]]
        example["input_ids"] = tokenizer(prompt, padding="max_length",truncation=True).input_ids
        example["labels"] = tokenizer(example["answer"], padding="max_length", truncation=True).input_ids
        return example
    
    tokenized_dataset = dataset.map(tokenize_example, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["question","answer"])
    return tokenized_dataset