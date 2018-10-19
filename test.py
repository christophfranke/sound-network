import keras
import numpy as np
import sounddevice as sd
import math
import random
from functools import reduce
import matplotlib.pyplot as plt


from keras import models
from keras.layers import Dense, LeakyReLU, Activation, Conv1D, Flatten, LSTM, Reshape

input_count = 1024

model = models.Sequential()
# model.add(Conv1D(64, kernel_size=2, strides=2, input_shape=(input_count, 1)))
# model.add(Conv1D(32, kernel_size=2, strides=4))
# model.add(Conv1D(32, kernel_size=2, strides=2))
# model.add(Conv1D(32, kernel_size=2, strides=2))
# model.add(Conv1D(16, kernel_size=1, strides=2))
# model.add(Flatten())
# model.add(Reshape((64, 1)))
# model.add(LSTM(64))
# model.add(Dense(50, activation='relu', input_shape=(input_count, 1)))
#model.add(Dense(1, activation='relu', input_shape=(input_count, 1)))
model.add(Flatten(input_shape=(input_count, 1)))
# model.add(Dense(64, activation='relu'))
#model.add(Dense(512, activation='relu'))
#model.add(Dense(256, activation='relu'))
model.add(Dense(input_count, activation='linear'))


# For a mean squared error regression problem
model.compile(optimizer='adam',
              loss='mse')

model.summary()

def sanitize(x):
  return float(1 if x > 1 else -1 if x < -1 else x)

def sanitize_sequence(l):
  return list(map(sanitize, l))

def sin(time):
  return math.sin(2*math.pi * time)

def square(time):
  frame = time % 2*math.pi
  return float(1) if frame < math.pi else float(-1)

def saw(time):
  frame = time % 2*math.pi
  return float(frame / math.pi - 1)

#oszilators = [sin, square, saw]
oszilators = [sin]

# Generate dummy data
duration = 1  # seconds
fs = 44100
samples = int(fs * duration)
#base_pitch = 220 #hz
#relative_pitch = 0*base_pitch
data = np.array([i/fs for i in range(samples)])

train_x = []
train_y = []

def sample(pitch, offset, count, osz, amplitude = 1):
  global fs
  return [amplitude*osz(pitch * (i + offset) / fs) for i in range(count)]

def sum_samples(*samples):
  return [sanitize(sum([sample[i] for sample in samples]) / len(samples)) for i in range(len(samples[0]))]

for i in range(samples):
  pitch = 220
  offset = np.random.rand() * fs
  oszilator = random.choice(oszilators)
  amplitude = 1

  x1 = sample(pitch, offset, input_count, oszilator, amplitude)
  y1 = sample(pitch, offset + input_count, input_count, oszilator, amplitude)

  pitch = 329.63
  offset = np.random.rand() * fs
  oszilator = random.choice(oszilators)
  amplitude = 1

  x2 = sample(pitch, offset, input_count, oszilator, amplitude)
  y2 = sample(pitch, offset + input_count, input_count, oszilator, amplitude)

  pitch = 554.37
  offset = np.random.rand() * fs
  oszilator = random.choice(oszilators)
  amplitude = 1

  x3 = sample(pitch, offset, input_count, oszilator, amplitude)
  y3 = sample(pitch, offset + input_count, input_count, oszilator, amplitude)

  train_x.append(sum_samples(x1, x2, x3))
  train_y.append(sum_samples(y1, y2, y3))

# plt.plot(range(input_count), train_x[0])
# plt.plot(range(input_count), train_x[1])
# plt.plot(range(input_count), train_x[2])
# plt.show()

train_x = np.array(train_x).reshape(samples, input_count, 1)
train_y = np.array(train_y)

def generate(model):
  global fs, input_count

  length = 100

  pitch = 220
  offset = np.random.rand() * fs
  oszilator = random.choice(oszilators)
  amplitude = 1
  x1 = sample(pitch, offset, input_count, oszilator, amplitude)
  y1 = sample(pitch, offset, input_count * length, oszilator, amplitude)

  pitch = 329.63
  offset = np.random.rand() * fs
  oszilator = random.choice(oszilators)
  amplitude = 1
  x2 = sample(pitch, offset, input_count, oszilator, amplitude)
  y2 = sample(pitch, offset, input_count * length, oszilator, amplitude)

  pitch = 554.37
  offset = np.random.rand() * fs
  oszilator = random.choice(oszilators)
  amplitude = 1
  x3 = sample(pitch, offset, input_count, oszilator, amplitude)
  y3 = sample(pitch, offset, input_count * length, oszilator, amplitude)

  input_data = sum_samples(x1, x2, x3)
  correct_sound = sum_samples(y1, y2, y3)

  sound_data = []
  for i in range(length):
    prediction = model.predict(np.array([input_data]).reshape(1, input_count, 1))[0]
    input_data = sanitize_sequence(prediction)
    sound_data = sound_data + input_data

  return sound_data, correct_sound

while True:
  model.fit(train_x, train_y, epochs=1, batch_size=32)
  sound_data, correct_sound = generate(model)
  # plt.plot(range(len(sound_data) + len(input_data)), input_data + sound_data)
  model.save('last_model')
  # plt.plot([i + len(input_data) for i in range(len(sound_data))], sound_data)
  # plt.plot(range(len(input_data)), input_data)
  # plt.show()
  #sd.play(sound_data + sound_data + sound_data + sound_data + sound_data)
  sd.play(correct_sound + sound_data)

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