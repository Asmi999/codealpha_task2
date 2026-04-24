#!/usr/bin/env python
# coding: utf-8

# In[ ]:


installing libraries


# In[15]:


get_ipython().system('pip install librosa numpy pandas scikit-learn tensorflow')

iimporting libraries
# In[65]:


import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


# In[ ]:


feature extraction


# In[66]:


def extract_features(file_path):
    import librosa
    import numpy as np
    audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)


# In[ ]:


feature test file


# In[18]:


test_file = "dataset/Actor_01/03-01-01-01-01-01-01.wav"

features = extract_features(test_file)

print(features)
print("Shape:", features.shape)


# In[ ]:


load dataset


# In[67]:


dataset_path = "dataset"

features = []
labels = []

for actor in os.listdir(dataset_path):
    actor_path = os.path.join(dataset_path, actor)

    for file in os.listdir(actor_path):
        if file.endswith(".wav"):
            file_path = os.path.join(actor_path, file)

            emotion = file.split("-")[2]

            feature = extract_features(file_path)
            features.append(feature)
            labels.append(emotion)


# In[68]:


emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

features = []
labels = []

for actor in os.listdir(dataset_path):
    actor_path = os.path.join(dataset_path, actor)

    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):
        if file.endswith(".wav"):
            file_path = os.path.join(actor_path, file)

            emotion_code = file.split("-")[2]
            emotion = emotion_dict[emotion_code]

            feature = extract_features(file_path)
            features.append(feature)
            labels.append(emotion)


# In[ ]:


preprocessing code


# In[69]:


import numpy as np   # ← add this

X = np.array(features)
y = np.array(labels)

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

le = LabelEncoder()
y_encoded = to_categorical(le.fit_transform(y))


# In[ ]:


step1 for model building


# In[70]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)


# In[ ]:


step2 for model building


# In[71]:


X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)


# In[ ]:


model building code


# In[ ]:





# In[72]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(y_train.shape[1], activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[73]:


model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))


# In[ ]:


accuracy prediction


# In[74]:


loss, acc = model.evaluate(X_test, y_test)
print("Accuracy:", acc)


# In[75]:


history = model.fit(
    X_train, y_train,
    epochs=2,
    batch_size=32,
    validation_data=(X_test, y_test)
)


# In[76]:


print(le.classes_)


# In[ ]:


emotion prediction


# In[77]:


def predict_emotion(file_path):
    feature = extract_features(file_path)
    feature = np.expand_dims(feature, axis=0)
    feature = np.expand_dims(feature, axis=2)

    prediction = model.predict(feature, verbose=0)

    predicted_index = np.argmax(prediction)
    emotion = le.classes_[predicted_index]

    return emotion

print(predict_emotion("dataset/Actor_01/03-01-01-01-01-01-01.wav"))


# In[ ]:


final output


# In[78]:


def predict_emotion_with_confidence(file_path):
    print("Function started")   # 👈 debug

    feature = extract_features(file_path)
    feature = np.expand_dims(feature, axis=0)
    feature = np.expand_dims(feature, axis=2)

    prediction = model.predict(feature, verbose=0)

    predicted_index = np.argmax(prediction)
    emotion = le.classes_[predicted_index]
    confidence = np.max(prediction)

    print("Emotion:", emotion)
    print("Confidence:", round(float(confidence)*100, 2), "%")


# In[79]:


predict_emotion_with_confidence("dataset/Actor_01/03-01-01-01-01-01-01.wav")

  





REAL TIME EXAMPLE


# In[38]:


get_ipython().system('pip install sounddevice scipy')


# In[80]:


import sounddevice as sd
from scipy.io.wavfile import write
import os

def record_audio(filename="test.wav", duration=3, fs=22050):
    if os.path.exists(filename):
        os.remove(filename)   # 🔥 delete old file

    print("🎤 Recording... Speak now!")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)
    print("✅ Recording saved!")


# In[81]:


def predict_from_mic():
    record_audio("test.wav", duration=3)   # 🎤 record

    import time
    time.sleep(1)   # ensure file is saved

    print("Processing...")

    emotion = predict_emotion("test.wav")

    print("Predicted Emotion:", emotion)


# In[82]:


predict_from_mic()






