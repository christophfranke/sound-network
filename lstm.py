import keras
import numpy as np
import sounddevice as sd
import math
import random
from functools import reduce
import matplotlib.pyplot as plt


from keras import models
from keras.layers import Dense, LeakyReLU, Activation, Conv1D, Flatten, LSTM, Reshape

input_count = 64

model = models.Sequential()
model.add(LSTM(input_count, input_shape=(input_count, 1)))
# model.add(Reshape(target_shape=(1, input_count)))
# model.add(LSTM(input_count))
model.add(Dense(input_count, activation='relu'))
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

oszilators = [sin, square, saw]
#oszilators = [sin]

# Generate dummy data
duration = 20.0 / input_count  # seconds
fs = 44100
samples = int(fs * duration) * input_count
#base_pitch = 220 #hz
#relative_pitch = 0*base_pitch
data = np.array([i/fs for i in range(samples)])

def sample(pitch, offset, count, osz, amplitude = 1):
  global fs
  return [amplitude*osz(pitch * (i + offset) / fs) for i in range(count)]

def sum_samples(*samples):
  return [sanitize(sum([sample[i] for sample in samples]) / len(samples)) for i in range(len(samples[0]))]

pitch = 220
offset = np.random.rand() * fs
oszilator = random.choice(oszilators)
amplitude = 1

x1 = sample(pitch, offset, samples, oszilator, amplitude)
y1 = sample(pitch, offset + samples, samples, oszilator, amplitude)

pitch = 329.63
offset = np.random.rand() * fs
oszilator = random.choice(oszilators)
amplitude = 1

x2 = sample(pitch, offset, samples, oszilator, amplitude)
y2 = sample(pitch, offset + samples, samples, oszilator, amplitude)

pitch = 554.37
offset = np.random.rand() * fs
oszilator = random.choice(oszilators)
amplitude = 1

x3 = sample(pitch, offset, samples, oszilator, amplitude)
y3 = sample(pitch, offset + samples, samples, oszilator, amplitude)

# plt.plot(range(input_count), train_x[0])
# plt.plot(range(input_count), train_x[1])
# plt.plot(range(input_count), train_x[2])
# plt.show()

train_x = np.array(sum_samples(x1, x2, x3)).reshape(int(samples / input_count), input_count, 1)
train_y = np.array(sum_samples(y1, y2, y3)).reshape(int(samples / input_count), input_count)

def generate(model):
  global fs, input_count

  length = 1

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
  model.fit(train_x, train_y, epochs=1, batch_size=32, shuffle=False)
  # sound_data, correct_sound = generate(model)
  # plt.plot(range(len(sound_data) + len(input_data)), input_data + sound_data)
  model.save('last_model')
  # plt.plot([i + len(input_data) for i in range(len(sound_data))], sound_data)
  # plt.plot(range(len(input_data)), input_data)
  # plt.show()
  #sd.play(sound_data + sound_data + sound_data + sound_data + sound_data)
  #sd.play(correct_sound + sound_data)

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