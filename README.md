# 🧠 LLM Labs – PyTorch for Large Language Models

This repository contains Python labs for the **"Training and Parallelization of Large Language Models (LLMs)"** course, offered at the Arab Academy for Science, Technology & Maritime Transport (AASTMT) in collaboration with Virginia Tech (Summer 2025).

## 🔍 About the Course
This course explores how to train, optimize, and parallelize large language models (LLMs) using techniques like:
- Data, model, pipeline, and tensor parallelism
- Distributed training
- Fine-tuning and benchmarking TinyLlama
- Efficient scaling and resource usage

## 🧪 Labs

### 📘 Lab 1: Introduction to PyTorch
Topics covered:
- What is PyTorch?
- Tensor operations and autograd
- Training a simple linear regression model
- Classwork for hands-on practice

📄 [Open Lab 1 Notebook](Lab_1_intro_to_pytorch.ipynb)

---

### 📘 Lab 2: MLP for Multi-class Classification
Topics covered:
- Difference between classification vs. regression
- Binary vs. multi-class classification
- Using `sklearn.datasets.load_digits` (Mini MNIST)
- Building an MLP classifier from scratch
- Using PyTorch’s trainer tools
- Visualizing training and predictions

📄 [Open Lab 2 Notebook](Lab_2_MLP_for_muliclassfication.ipynb)

---
## 📚 Lab 3: Tokenization and Embeddings

Topics covered:
- Manual and subword tokenization (using HuggingFace Tokenizers)
- Vocabulary building for language models
- Embedding layers and how they represent tokens
- Training a basic next-word prediction model
- Visualizations for embeddings

📄 [Open Lab 3 Notebook](Lab_3_Tokenization_and_Embeddings_for_Language_Modeling.ipynb)

## ⚙️ Installation

To run the labs, install the following packages:

```bash
pip install torch torchvision torchaudio matplotlib scikit-learn transformers tokenizers seaborn
