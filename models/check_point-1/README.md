
# 🌌 Nova-Talk

**Nova-Talk** is a domain-adapted language model fine-tuned from [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base), designed to answer questions related to astrophysics. It uses **LoRA (Low-Rank Adaptation)** via the [PEFT](https://github.com/huggingface/peft) library for efficient fine-tuning with limited compute resources.

---

## 📊 Model Summary

- **Base Model**: `google/flan-t5-base`
- **Fine-tuning Strategy**: LoRA (Low-Rank Adaptation)
- **Library Used**: `peft`
- **Total Trainable Parameters**: `6,553,088` (approx.)
- **Compute Device**:  
  - **Laptop**: Lenovo IdeaPad Gaming 3  
  - **GPU**: NVIDIA RTX 3050 (4GB VRAM)

---

## ⚙️ Training Configuration

```python
learning_rate = 1e-4
num_train_epochs = 250
max_steps = 25000
```

---

## 🔧 LoRA Configuration

```python
LoraConfig(
    r = 32,
    lora_alpha = 32,
    target_modules = ["q", "v"],
    lora_dropout = 0.1,
    bias = "none",
    task_type = TaskType.SEQ_2_SEQ_LM
)
```


---

## 📝 Notes

> ⚠️ **This model is not fully fine-tuned** due to computational resource limitations. It is currently a partially-trained prototype and will be further fine-tuned in the future as resources allow.

---

## 🚀 Coming Soon

- 🌐 Web-based chatbot deployment  
- 📚 Training on a larger astrophysics corpus  
- 📈 Evaluation metrics and benchmark reports  
- 🤖 Hugging Face model card and demo

---

## 📁 Repository Info

This repository contains:
- Fine-tuning scripts and LoRA configuration
- Dataset preprocessing code
- Notebook logs and training metadata
- Initial version of the fine-tuned Nova-Talk model

---

Feel free to fork or contribute to the project!