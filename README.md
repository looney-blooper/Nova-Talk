# 🌌 NovaTalk

NovaTalk is an intelligent chatbot designed to answer questions related to **astrophysics**. It is powered by a fine-tuned [FLAN-T5 base model](https://huggingface.co/google/flan-t5-base) using the [AstrophysicsQuestionAndAnswer dataset](https://huggingface.co/datasets/LazySloth/AstrophysicsQuestionAndAnswer) from Hugging Face, curated by LazySloth.

---

## 🚀 Overview

NovaTalk leverages the capabilities of the FLAN-T5 base model fine-tuned on domain-specific knowledge in astrophysics to provide accurate and contextually relevant answers to complex scientific queries. It can be used for educational assistance, academic exploration, or as a conversational science assistant.

---

## 🧠 Model Details

- **Base Model**: [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base)
- **Fine-Tuned On**: [`LazySloth/AstrophysicsQuestionAndAnswer`](https://huggingface.co/datasets/LazySloth/AstrophysicsQuestionAndAnswer)
- **Task**: Question Answering (QA)
- **Language**: English
- **Domain**: Astrophysics / Space Science

> ⚠️ **Note:** Model training is currently in progress and has not yet reached its optimal state due to limited computational resources. Performance will improve as training continues or is migrated to more powerful hardware.

---

## 📚 Dataset Description

The dataset used is a collection of question-answer pairs focused on astrophysics topics, compiled and structured to support natural language understanding and generation tasks.

- **Source**: Hugging Face
- **URL**: [https://huggingface.co/datasets/LazySloth/AstrophysicsQuestionAndAnswer](https://huggingface.co/datasets/LazySloth/AstrophysicsQuestionAndAnswer)

---

## 🛠 Installation & Setup

1. **Clone this repository**:
   ```bash
   git clone https://github.com/yourusername/NovaTalk.git
   cd NovaTalk

2. **Install the Required Libraries**
   ```bash
   pip install -r requirements.txt  #For Fine tuning the model

                                 **OR**

   ```bash
   pip install transformers torch  #For only model inference

###Fine tuning the Model
1. **Run the main.py file**
   ```bash
   python main.py   #REMEMBER ... This step may take long time to execute as the model is fine tuning from the scratch.

###To interact with the fine tuned model
1. **Run the Nova-talk-askme.py file**
   ```bash
   python nova_talk_askme.py