import keras
import numpy as np
import sounddevice as sd
import math
import random
import matplotlib.pyplot as plt


from keras import models
from keras.layers import Dense, LeakyReLU, Activation, Conv1D, Flatten

input_count = 250

model = models.Sequential()
model.add(Dense(10, activation='relu', input_shape=(input_count,)))
model.add(Dense(1000, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='linear'))


# For a mean squared error regression problem
model.compile(optimizer='adam',
              loss='mse')

model.summary()

def sanitize(x):
  return float(1 if x > 1 else -1 if x < -1 else x)

def sin(time):
  return math.sin(2*math.pi * time)

def square(time):
  frame = time % 2*math.pi
  return float(1) if frame < math.pi else float(-1)

def saw(time):
  frame = time % 2*math.pi
  return float(frame / math.pi - 1)

oszilators = [sin, square, saw]

# Generate dummy data
duration = 5  # seconds
fs = 44100
samples = int(fs * duration)
base_pitch = 110 #hz
relative_pitch = 8*base_pitch
data = np.array([i/fs for i in range(samples)])

train_x = []
train_y = []

def sample(pitch, offset, count, osz):
  global fs
  return [osz(pitch * (i + offset) / fs) for i in range(count)]

def sum_samples(sample1, sample2):
  return [sanitize(0.6*sample1[i] + 0.6*sample2[i]) for i in range(len(sample1))]

for i in range(samples):
  pitch = base_pitch + np.random.rand() * relative_pitch
  offset = np.random.rand() * fs
  oszilator = random.choice(oszilators)

  x1 = sample(pitch, offset, input_count, oszilator)
  y1 = sample(pitch, offset + input_count, 1, oszilator)

  pitch = base_pitch + np.random.rand() * relative_pitch
  offset = np.random.rand() * fs
  oszilator = random.choice(oszilators)

  x2 = sample(pitch, offset, input_count, oszilator)
  y2 = sample(pitch, offset + input_count, 1, oszilator)

  train_x.append(sum_samples(x1, x2))
  train_y.append(sum_samples(y1, y2))

# plt.plot(range(input_count), train_x[0])
# plt.plot(range(input_count), train_x[1])
# plt.plot(range(input_count), train_x[2])
# plt.show()

train_x = np.array(train_x)
train_y = np.array(train_y)

def generate(model):
  global fs, input_count

  pitch = base_pitch + np.random.rand() * relative_pitch
  offset = np.random.rand() * fs
  oszilator = random.choice(oszilators)
  x1 = sample(pitch, offset, input_count, oszilator)

  pitch = base_pitch + np.random.rand() * relative_pitch
  offset = np.random.rand() * fs
  oszilator = random.choice(oszilators)
  x2 = sample(pitch, offset, input_count, oszilator)

  input_data = sum_samples(x1, x2)

  sound_data = []
  initial_input_data = [x for x in input_data]
  length = int(0.25*fs)
  length = 3 * input_count
  for i in range(length):
    prediction = sanitize(model.predict(np.array([input_data]))[0][0])
    input_data.append(prediction)
    input_data.pop(0)
    sound_data.append(prediction)

  return sound_data, initial_input_data

while True:
  model.fit(train_x, train_y, epochs=1, batch_size=32)
  sound_data, input_data = generate(model)
  plt.plot(range(len(sound_data) + len(input_data)), input_data + sound_data)
  model.save('last_model')
  # plt.plot([i + len(input_data) for i in range(len(sound_data))], sound_data)
  # plt.plot(range(len(input_data)), input_data)
  plt.show()
  #sd.play(sound_data + sound_data + sound_data + sound_data + sound_data)
  #sd.play(sound_data)

# input_data = np.array([i/fs for i in range(samples)])
# prediction = model.predict(input_data).reshape(samples)
# print(min(prediction), max(prediction))

# startTime = 0
# def callback(outdata, frames, time, status):
#   global startTime
#   if status:
#     print(status)
#   if startTime == 0:
#     startTime = time.outputBufferDacTime
#   current_time = time.outputBufferDacTime - startTime
#   input_data = np.array([i/fs + current_time for i in range(frames)])
#   prediction = model.predict(input_data).reshape(frames)
#   data = np.array([prediction, prediction])
#   outdata[:] = np.transpose(data)

# with sd.OutputStream(channels=2, callback=callback):
#   sd.sleep(int(10 * duration * 1000))