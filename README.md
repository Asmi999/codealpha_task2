# 🎤 Speech Emotion Recognition

A machine learning project that detects **human emotions from speech audio** using deep learning and signal processing techniques.

---

## 📌 Overview

This system analyzes speech audio and predicts the **underlying emotion** such as *happy, sad, angry,* etc.
It uses **MFCC (Mel-Frequency Cepstral Coefficients)** for feature extraction and a **deep learning model (LSTM/CNN)** for classification.

---

## 🚀 Features

* 🎧 Emotion prediction from audio files (.wav)
* 🧠 Deep learning-based classification (LSTM/CNN)
* 📊 MFCC feature extraction using Librosa
* 🎤 Real-time emotion detection using microphone input
* 📈 Confidence score for predictions

---

## 🛠️ Tech Stack

* Python
* NumPy
* Librosa
* TensorFlow / Keras
* Scikit-learn

---

## 📂 Project Structure

```
emotion_recognition/
│
├── emotion_recognition.ipynb   # Main notebook
├── emotion_recognition.py      # Converted script (optional)
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
```

---

## 📊 Dataset

This project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.

⚠️ Dataset is not included due to size.
👉 Download from Kaggle: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

---

## ⚙️ Installation

1. Clone the repository:

```
git clone https://github.com/your-username/Speech-Emotion-Recognition.git
cd Speech-Emotion-Recognition
```

2. Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶️ Usage

### 🔹 Run Notebook

Open and run:

```
emotion_recognition.ipynb
```

---

### 🔹 Predict from Audio File

```python
predict_emotion("path_to_audio.wav")
```

---

### 🔹 Real-Time Prediction (Microphone)

```python
predict_from_mic()
```

---

## 🎯 Output

The model predicts one of the following emotions:

* Angry 😠
* Calm 😌
* Happy 😊
* Sad 😢
* Fearful 😨
* Disgust 🤢
* Surprised 😲
* Neutral 😐

Example:

```
Emotion: happy
Confidence: 82.4%
```

---

## 📈 Model Performance

* Accuracy: ~70–80% (depends on training)
* Uses MFCC features + Deep Learning

---

## 💡 Future Improvements

* Improve accuracy with larger datasets
* Add real-time streaming (continuous listening)
* Build web app interface using Streamlit
* Deploy as a mobile/web application

---

## 🤝 Contribution

Feel free to fork this repository and improve the project.

---

## 📜 License

This project is for educational purposes.

---

## 👩‍💻 Author

Asmitha
GitHub: https://github.com/Asmi999
