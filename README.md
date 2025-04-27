# Federated Devanagari Letter Classifier

A Streamlit web app that predicts handwritten Devanagari letters (अ, आ, क, ख, etc.) using a **Federated Learning-trained CNN**.  
This project demonstrates secure on-device training while achieving high accuracy without centralizing data.

---

## 🚀 Live Site

[**Visit the live App**](https://your-streamlit-deployment-link.com)

---

## ✨ Features

- Upload your own **handwritten Devanagari image** or select a **sample**.
- **Federated CNN model** trained across distributed devices.
- Displays **predicted letter** and **confidence** instantly.
- Clean, responsive **UI with light theme**.
- Explore detailed **About** and **Why Federated?** tabs.
- View **example predictions**, implementation diagrams, and videos.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/federated-devanagari-classifier.git
cd federated-devanagari-classifier
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Sample `requirements.txt`:

```
streamlit
tensorflow
opencv-python
numpy
```

### 3. Run the App

```bash
streamlit run app.py
```

---

## 📋 Model Overview

- **Input:** 32×32 grayscale images of Devanagari characters.
- **Classes:** 46 (36 letters + 10 numerals)
- **Architecture:**
  - 4× Conv2D → BatchNorm → MaxPooling
  - Dense(128) → BatchNorm → Dropout
  - Dense(64) → BatchNorm → Dropout
  - Dense(46, Softmax)
- **Federated Averaging** used across 3 clients for training.

---

## Why Federated Learning?

- **Data Privacy:** No raw data leaves devices.
- **Speed:** Faster training compared to centralized learning.
- **Robustness:** Learns diverse handwriting styles.
- **Scalability:** Easily add more edge devices.
- **Security:** Protects personal handwriting samples.

---

## 📸 Screenshots

[Home](assets/homepage_screenshot.png)

[Prediction](assets/prediction_screenshot.png) |

---
