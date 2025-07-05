from datasets import Dataset, load_dataset

def get_dataset(dataset_name,split="train"):
    dataset = load_dataset(dataset_name)
    dataset = Dataset.from_list(dataset[split]["qa_pairs"][0])
    return dataset