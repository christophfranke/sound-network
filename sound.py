import sounddevice as sd
import numpy as np

import sounddevice as sd
duration = 2  # seconds

def callback(outdata, frames, time, status):
  if status:
    print(status)

  data = np.array([np.random.rand(frames), np.random.rand(frames)])
  outdata[:] = np.transpose(data)

with sd.OutputStream(channels=2, callback=callback):
  sd.sleep(int(duration * 1000))