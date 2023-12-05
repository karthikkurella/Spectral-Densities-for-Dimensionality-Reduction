import os
from numba import jit
import numpy as np
from scipy.io import wavfile
import scipy.io


import numpy as np
from scipy.stats import skew, kurtosis, moment
from scipy.fft import fft, ifft


# Assuming 'x' and 'Fbe' are already defined, and 'Fs' is the sampling frequency
# Make sure 'x' is a numpy array (e.g., x = np.array([...]))

import numpy as np
import scipy.io
import scipy.signal
import scipy.ndimage
from numba import jit

# The octspace function is not directly available in Python's standard libraries, you may have to implement it
# Fbe = octspace(10, 6000, 12).center  # Your octave space function
Fbe = np.logspace(np.log10(10), np.log10(6000), 120)  # Placeholder using logspace
Fbe = Fbe  # Assuming Fbe.center in MATLAB just returns Fbe

@jit(nopython=True)
def numba_fft(x):
    N = len(x)
    X = np.empty(N, dtype=np.complex128)
    for k in range(N):
        X[k] = 0.0j
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X
@jit(nopython=True)
def loychikSD_NP(x, Fs, Fbe):
    # Frequency bin centers
    F = Fbe[:-1] + np.diff(Fbe)

    X = numba_fft(x)
    t = np.arange(0, len(x)) / Fs  # equivalent to the original t calculation
    dF = 1 / t[-1]
    Ff = np.arange(0, len(x)) * dF

    # Statistical moments
    M = np.array([
        np.mean(x),
        np.var(x),
        skew(x),
        kurtosis(x),
        moment(x, 1),
        moment(x, 2),
        moment(x, 3)
    ])

    # Initialize result arrays
    Gx = np.zeros(len(F))
    Sx = np.zeros(len(F))
    Kx = np.zeros(len(F))
    M3x = np.zeros(len(F))
    M4x = np.zeros(len(F))
    
    #Kx = M4x/(Gx)^2

    for ii in range(len(F)):
        if(ii%5==0):
            print(f"{ii + 1} / {len(F)}")  # Counter to determine where you are in run

        # Begin low-pass filter by zeroing bins
        Xn = np.argmax(Ff >= Fbe[ii + 1])  # find the first index where Ff >= Fbe[ii + 1]

        cutX = X[1:Xn]
        XI = np.zeros_like(X)
        XI[1:Xn] = cutX

        # Flipping and conjugating for symmetry
        XI = np.flipud(XI)
        XI[0:Xn - 1] = np.conj(cutX)
        XI = np.flipud(XI)

        xifft = ifft(XI)

        # Extracting the real part of the inverse FFT result
        real_xifft = np.real(xifft)  # this makes sure you're calculating statistics on real numbers

        # Statistical calculations
        Gx[ii] = np.var(real_xifft)
        Sx[ii] = skew(real_xifft)
        Kx[ii] = kurtosis(real_xifft)+3
        M3x[ii] = moment(real_xifft, 3)
        M4x[ii] = moment(real_xifft, 4)

    return M, F, Gx, Sx, Kx

# The variables Gx, Sx, Kx, M3x, M4x hold the results and can be used as needed



# Function to process a single audio file
def process_audio_file(fp, idx):
    with open(fp, 'rb') as wf:
        Fs, audio_data = wavfile.read(wf)
    x3 = audio_data.astype(float)

    M, F, Gx, Sx, Kx = loychikSD_NP(x3, Fs, Fbe)

    # Power Spectral Density
    Gxx = np.gradient(Gx, F)
    # Skewness Spectral Density
    Sxx = np.gradient(Sx, F)
    # Kurtosis Spectral Density
    Kxx = np.gradient(Kx, F)
    
    middle_c_freq = 261.63
    tolerance = 5 
    middle_c_peak = np.where((F > middle_c_freq - 160) & (F < middle_c_freq + 1100))
    
    Gx = Gx[middle_c_peak[0]]
    Sx = Sx[middle_c_peak[0]]
    Kx = Kx[middle_c_peak[0]]
    Gxx = Gxx[middle_c_peak[0]]
    Sxx = Sxx[middle_c_peak[0]]
    Kxx = Kxx[middle_c_peak[0]]
    
    # Specify the output path
    output_path = '/Users/karthikkurella/Documents/Audio_mat/'
    output_filename = f"{filename.split('_')[0]}-{idx + 1}.mat"

    mat_data = {
        "M": M,
        "F": F,
        "Gx": Gx,
        "Sx": Sx,
        "Kx": Kx,
        "Gxx": Gxx,
        "Sxx": Sxx,
        "Kxx": Kxx
    }

    # Save the dictionary into a .mat file
    scipy.io.savemat(os.path.join(output_path, output_filename), mat_data)

    print(f"Saved {output_filename}")

# Parallel processing function
def parallel_process_files(file_list):
    for idx, file in enumerate(file_list):
        print(f"Processing {file} (File {idx + 1}/{len(file_list)})")
        process_audio_file(file, idx)

def process_with_args(args):
    file, idx = args
    partial_process_audio_file(file, idx)