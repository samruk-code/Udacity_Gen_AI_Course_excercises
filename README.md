# Udacity Generative AI Course — Exercises & Projects

A collection of hands-on notebooks from Udacity's Generative AI course, covering foundational deep learning with PyTorch, transfer learning, and NLP with HuggingFace Transformers.

---

## Projects

### 1. Simple PyTorch Neural Network
**[`PyTorch_simple_neural_network.ipynb`](PyTorch_simple_neural_network.ipynb)**

Builds a full PyTorch pipeline from scratch — custom `Dataset`, `DataLoader`, MLP architecture, training loop, and inference. The model learns to predict the sum of two numbers.

**Topics:** Custom Dataset, DataLoader, MLP, MSELoss, Adam optimizer

---

### 2. MNIST Handwritten Digit Classification (MLP)
**[`Classification-of-handwritten-digits-using-an-MLP.ipynb`](Classification-of-handwritten-digits-using-an-MLP.ipynb)**

Trains an MLP classifier on the MNIST dataset (60,000 training images) using scikit-learn. Achieved **97% test accuracy** and **98.7% training accuracy**.

**Topics:** sklearn `MLPClassifier`, data normalization, model evaluation, prediction visualization

---

### 3. Transfer Learning with MobileNetV3
**[`Exercise3-transfer-learning-using-mobilenetv3.ipynb`](Exercise3-transfer-learning-using-mobilenetv3.ipynb)**

Applies transfer learning using a pre-trained MobileNetV3 (Small) model on the Fashion-MNIST dataset. Freezes convolutional layers and replaces the final classifier head with a 10-class output layer. Achieved **85.52% test accuracy** after a single training epoch.

**Topics:** Transfer learning, feature freezing, torchvision models, Fashion-MNIST, Adam optimizer

---

### 4. GenAI Sentiment Classification
**[`GenAI_sentiment_classification.ipynb`](GenAI_sentiment_classification.ipynb)**

Sentiment classification project applying generative AI techniques to text analysis.

**Topics:** Sentiment analysis, NLP, classification

---

### 5. PyTorch & HuggingFace Scavenger Hunt
**[`Exercise2_pytorch_and_hugging_face_scavenger_huntscavenger_hunt.ipynb`](Exercise2_pytorch_and_hugging_face_scavenger_huntscavenger_hunt.ipynb)**

A guided exploration of PyTorch and HuggingFace:
- PyTorch tensors, MLP construction with `torch.nn`, loss functions, optimizers, and training loops
- Sentiment analysis using `distilbert-base-uncased-finetuned-sst-2-english` from HuggingFace
- Running inference on the IMDB movie reviews dataset

**Topics:** PyTorch fundamentals, HuggingFace Transformers, DistilBERT, IMDB dataset, AutoModel/AutoTokenizer

---

### 6. PyTorch Tutorials
**[`Pytorch_tutorials.ipynb`](Pytorch_tutorials.ipynb)**

Reference notebook covering core PyTorch building blocks with code examples:
- Tensor creation and manipulation
- Building MLPs with `nn.Module`
- Loss functions: `CrossEntropyLoss`, `MSELoss`
- Optimizers: SGD, Adam
- Custom `Dataset` and `DataLoader`

---

### 7. Exercise 1 — MNIST MLP (Guided)
**[`Exercise1-classification-of-handwritten-digits-using-an-mlp.ipynb`](Exercise1-classification-of-handwritten-digits-using-an-mlp.ipynb)**

Follow-along exercise for building an MLP classifier on MNIST using scikit-learn. Walks through data loading, model definition, training, evaluation, and prediction visualization.

---

## Tech Stack

| Library | Usage |
|---|---|
| PyTorch | Neural networks, training loops, custom datasets |
| torchvision | Pre-trained models (MobileNetV3), Fashion-MNIST |
| HuggingFace Transformers | DistilBERT sentiment analysis |
| HuggingFace Datasets | IMDB dataset loading |
| scikit-learn | MLPClassifier, MNIST loading |
| TensorFlow/Keras | MNIST data loading |
| Matplotlib | Visualization |

## Setup

```bash
pip install torch torchvision transformers datasets scikit-learn matplotlib numpy
```
