#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read,write
from IPython.display import Audio
from numpy.fft import fft, ifft
import scipy
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import os

file_lists = []

filepath = '/Users/karthikkurella/Documents/MATLAB/Examples/R2023a/supportfiles/audio/AirCompressorDataset/AirCompressorDataset/Healthy/'
for filename in os.listdir(filepath):
    if filename.endswith('.wav'):
        file_lists.append(filename)

file_lists.sort()
# get the folders in the list
len(file_lists)


# In[3]:


# list the files
file_lists[:10]


# In[6]:


import wave
import numpy as np

# Function to concatenate WAV files
def concatenate_wav_files(input_files, output_file):
    # Create an empty list to store audio data
    audio_data = []

    # Iterate through the input WAV files
    for input_file in input_files:
        filepath = '/Users/karthikkurella/Documents/MATLAB/Examples/R2023a/supportfiles/audio/AirCompressorDataset/AirCompressorDataset/Healthy'
        with wave.open(filepath+'/'+input_file, 'rb') as wf:
            # Read audio data and append to the list
            audio_data.append(np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16))

    # Concatenate audio data
    concatenated_audio = np.concatenate(audio_data)

    # Create a new WAV file for the concatenated audio
    with wave.open(output_file, 'wb') as wf:
        wf.setparams((1, 2, 44100, 0, 'NONE', 'not compressed'))
        wf.writeframes(concatenated_audio.tobytes())

# List of input WAV files
input_files = file_lists  # Add your file names here

# Output WAV file
output_file = 'output_225.wav'

# Call the function to concatenate the WAV files
concatenate_wav_files(input_files, output_file)


# In[7]:


# Get the frequencies

# check the file

output_file = '/Users/karthikkurella/Downloads/output_225.wav'
Fs, data = read(output_file)

frequencies = fft(data)
print(frequencies)
N = len(data)

# Calculate the frequency values
freq_values = np.fft.fftfreq(N, 1 / Fs)
# Find the index of the maximum amplitude in the FFT result
dominant_freq_index = np.argmax(np.abs(frequencies))

# Get the dominant frequency in Hz
dominant_frequency = frequencies[dominant_freq_index]
print("Dominant Frequency:", dominant_frequency, "Hz")


# In[8]:


#listen to the Audio

Audio(data,rate=Fs)


# In[9]:


# get the length of the audio file

length = data.shape[0] / Fs
print(f"length = {length}s")


# In[10]:


time = np.linspace(0., length, data.shape[0])
plt.plot(time, data[:])
#plt.plot(time, data[:, 1], label="Right channel")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()

