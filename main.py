from scripts.get_model_tokenizer import get_model_tokenizer
from data.get_dataset import get_dataset
from data.tokenize_dataset import tokenize_dataset
from scripts.train import train_model
from scripts.plot_loss_graph import plot_loss_graph

base_model_name = "google/flan-t5-base"
dataset_name = "LazySloth/AstrophysicsQuestionAndAnswer"


orginal_model, tokenizer = get_model_tokenizer(base_model_name)
dataset = get_dataset(dataset_name,"train")

tokenized_dataset = tokenize_dataset(dataset,tokenizer)

fine_tuned_model_path = "./models/check_point-1"

model_training_history = train_model(orginal_model,tokenizer,tokenized_dataset,peft_model_path=fine_tuned_model_path)

#plot_loss_graph(model_training_history)

print("OVER")



