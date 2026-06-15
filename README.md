# 🧠 Handwritten Digit Recognition using CNN

A deep learning web application that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) built with TensorFlow and deployed using Streamlit.

---

## 🚀 Live Demo

🔗 **Live App:** https://codealphahandwrittencharacterrecognition.streamlit.app/

🔗 **GitHub Repository:** https://github.com/Kushagra25-hub/CodeAlpha_HandwrittenCharacterRecognition.git

---

## 📌 Project Overview

This project uses a Convolutional Neural Network (CNN) trained on the MNIST dataset to recognize handwritten digits in real time.

Users can draw a digit directly on an interactive canvas, and the model predicts the digit along with confidence scores and probability distribution.

---

## ✨ Features

* 🎨 Interactive drawing canvas
* 🔢 Real-time handwritten digit recognition
* 📊 Confidence score display
* 📈 Probability distribution visualization
* 🏆 Top 3 prediction results
* 📝 Prediction history tracking
* 🌐 Web-based interface using Streamlit
* 🚀 Deployed online for public access

---

## 🛠️ Technologies Used

### Frontend

* Streamlit
* Plotly

### Machine Learning

* TensorFlow
* Keras
* Convolutional Neural Networks (CNN)

### Programming Language

* Python

---

## 📂 Project Structure

```text
CodeAlpha_HandwrittenCharacterRecognition/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── models/
│   └── emnist_model.h5
│
└── .streamlit/
    └── config.toml
```

---

## 📊 Dataset

### MNIST Handwritten Digits Dataset

The model is trained on the MNIST dataset containing:

* 70,000 handwritten digit images
* 10 classes (0–9)
* 28×28 grayscale images
* Standard benchmark dataset for image classification

Dataset Source:

https://yann.lecun.com/exdb/mnist/

---

## 🧠 Model Architecture

The CNN architecture consists of:

* Convolution Layer (32 Filters)
* Max Pooling Layer
* Convolution Layer (64 Filters)
* Max Pooling Layer
* Flatten Layer
* Dense Layer (128 Neurons)
* Dropout Layer
* Output Layer (10 Classes)

---

## 📈 Model Performance

| Metric            | Value   |
| ----------------- | ------- |
| Training Accuracy | ~99%    |
| Test Accuracy     | ~98-99% |
| Dataset           | MNIST   |

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/CodeAlpha_HandwrittenCharacterRecognition.git

cd CodeAlpha_HandwrittenCharacterRecognition
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 🎯 How to Use

1. Open the web application.
2. Draw a digit (0–9) on the canvas.
3. Click the **Predict** button.
4. View:

   * Predicted digit
   * Confidence score
   * Top 3 predictions
   * Probability distribution chart

---

## 🌐 Deployment

The application is deployed using **Streamlit Community Cloud**.

**Note:** Since the application is hosted on Streamlit Community Cloud (Free Tier), the app may go to sleep after periods of inactivity. The first request may take a few seconds to wake the application.


---

## 🔮 Future Enhancements

* Handwritten Alphabet Recognition (A–Z)
* Mobile-Friendly UI
* Model Explainability Visualizations
* User Authentication
* Multiple Language Support
* Advanced CNN Architectures

---

## 👨‍💻 Author

**Kushagra Srivastava**

AI & Machine Learning Enthusiast

GitHub: https://github.com/Kushagra25-hub

---

## 📜 License

This project is created for educational and internship purposes.

© 2025 Kushagra Srivastava
