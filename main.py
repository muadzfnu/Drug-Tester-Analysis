import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler


train_labels = []
train_samples = []

# Contoh data:
# uji coba obat untuk individu dari umur 16 - 100
# ada 500 partisipan.
# 95 persen 16-55 -> 950 orang
# 5 persen 56-100 -> 50 orang
# ada side efek masing2 grup 50 persen

for i in range(50):
    # umur 16-55 dengan side effect
    random_younger = randint(16,55)
    train_samples.append(random_younger)
    train_labels.append(1)
    # umur 56-100 tanpa side effect
    random_older = randint(56,100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    # umur 16-55 tanpa side effect
    random_younger = randint(16,55)
    train_samples.append(random_younger)
    train_labels.append(0)
    # umur 56-100 dengan side effect
    random_older = randint(56,100)
    train_samples.append(random_older)
    train_labels.append(1)

# bu array into numpy format
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
# mengubah urutan data di train labels dan di train samples
train_labels, train_samples = shuffle(train_labels, train_samples)

# mengeskalasi data agar nilai data lebih kecil dan mudah untuk dihitung = efesiensi waktu
scaler = MinMaxScaler(feature_range=(0,1))
# mengubah data menjadi 2d data
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))

# # print data 
# for i in scaled_train_samples:
#     print(i)

# check GPU
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print("Num GPUs Available: ", len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# create model
model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax'),
])

# view model summary
model.summary()

#prepare model to train
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(x=scaled_train_samples, y=train_labels, validation_split=0.1, batch_size=10, epochs=30, shuffle=True, verbose=2)