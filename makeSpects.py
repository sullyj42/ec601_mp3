import numpy as np
from math import sin, pi
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import signal
import pywt
import scaleogram as scg

from time import sleep; 
# clc; clear all; close all; 
# addpath('/Users/JP-Macbook/Documents/MATLAB/Examples/R2019a/wavelet/FinancialDataExample')
# fs   = 8E3; t = 0:1/fs:1; 
# fs1  = 3E3; PRF1 = 4E-3 > mod(t, 1E-2);  y1 = sin(2*pi*fs1.*t).*PRF1; 
# fs2  = 50;  PRF2 = 6E-2 > mod(t, 5E-1);  y2 = sin(2*pi*fs2.*t).*PRF2;
# y = y1 + y2 ;%+ randn(1,length(y1)); 

# subplot(4,1,1); plot(t,y1,t,y2+2); title('Time Series Data');  legend('HF Signal', 'LF Signal'); 

# Sample signals
fs   = 8e3; 
T    = 1; 
N    = fs*T; 
t    = np.linspace(0, T, N)
# # Signal 1
fs1  = 3E3                     # High frequency, "fast" signal 
fs2  =  50
# y1 = np.array([0]); 
y2 = np.array([0]);
prf1 = (4E-3 > np.mod(t, 1E-2))
prf2 = (6E-2 > np.mod(t, 5E-1))
y1   = np.sin(2*pi*fs1*t)*prf1
y2   = np.sin(2*pi*fs2*t)*prf2
# for time in t:
#     prf1 = 4E-3 > (time % 1E-2)       # Apply modulating PRF
#     val1 = sin(2*pi*fs1*time)*prf1; 
#     np.append(y1, [val1])    # Hard-gated tone
#     prf2 = 6E-2 > (time % 5E-1) 
#     val2 = sin(2*pi*fs1*time)*prf2; 
#     np.append(y1, 2)    # Hard-gated tone

#     print(f'{val1:1.4f}   {val2:1.4f}')

print(f'DEBUG SHAPES: t: {t.shape}  prf1: {prf1.shape}    y1: {y1.shape}    y2: {y2.shape}')

print("trying to plot...\n")
plt.subplot(5, 1, 1)
plt.title("Time Series Data")
plt.plot(t, y1 - 1)
plt.plot(t, y2 + 1)
plt.xlabel("Time"); plt.ylabel("Amplitude")
Npts = 2**6 
NFFT = 2**7 # This is intuitively backwards

h = plt.subplot(5,1,2)
# h.set_yscale('log')
Pxx, freq, t2 = mlab.specgram(y1+y2, Fs = fs, NFFT = Npts, noverlap = round(Npts - Npts/3))
plt.title("High Time Resolution")
plt.xlabel("Time"); plt.ylabel("Frequency")
j = h.pcolor(t2, freq, 20*np.log10(Pxx), clim = [-100, -80])
j.set_clim(vmin = -100, vmax = -50)
# h.set_yscale('log')



# Spectrogram number 2
Npts = 2**11 
NFFT = 2**12 # This is intuitively backwards
h = plt.subplot(5,1,3)
Pxx, freq, t2 = mlab.specgram(y1+y2, Fs = fs, NFFT = Npts, noverlap = round(Npts - Npts/3))
plt.title("High Frequency Resolution")
plt.xlabel("Time"); plt.ylabel("Frequency")
j = h.pcolor(t2, freq, 20*np.log10(Pxx))
maxval = 20*np.log10(Pxx).max(); 
j.set_clim(vmin = maxval-40, vmax = maxval)
# h.set_yscale('log')
print('Computing wavelet')
widths = np.arange(2,31)
# Wavelet transform
scales = np.array([0.5333335])
print(scales); wind = 'gaus1'
freqs = pywt.scale2frequency(wind, scales)*fs;
print(f'Scales: {freqs}')
# '''
coef, freqs = pywt.cwt(y1+y2, sampling_period = 1/fs, scales = scales, wavelet = wind)
print('Plotting wavelet')
print(f'Shape of coef: {coef.shape}')
h =  plt.subplot(5,1,4)
j = h.pcolor(20*np.log10(abs(coef)+1e-8))
maxval = 20*np.log10(abs(coef)+1e-8).max()
j.set_clim(vmin = maxval-20, vmax = maxval)
plt.xlabel('Index of time (fix later)')
plt.ylabel('Index of frequency (fix later)                           \nSingle filterbank...                                  ')
plt.title('High Frequency wavelet filterbank')
scales = np.array([32])
print(scales); wind = 'gaus1'
freqs = pywt.scale2frequency(wind, scales)*fs;
print(f'Scales: {freqs}')
# '''
coef, freqs = pywt.cwt(y1+y2, sampling_period = 1/fs, scales = scales, wavelet = wind)
print('Plotting wavelet')
print(f'Shape of coef: {coef.shape}')
h =  plt.subplot(5,1,5)
j = h.pcolor(20*np.log10(abs(coef)+1e-8))
maxval = 20*np.log10(abs(coef)+1e-8).max()
plt.xlabel('Index of time (fix later)')
# plt.ylabel('Index of frequency (fix later)\nSingle filterbank...')
plt.title('Low Frequency wavelet filterbank')
j.set_clim(vmin = maxval-20, vmax = maxval)

# plt.colorbar(j);
# '''
'''
# # ATTEMPT with SciPy
# cwtmatrix = signal.cwt(y1+y2, signal.ricker, widths); 
'''

# ATTEMPT with Scaleogram
'''
scales = np.logspace(1, 10, num=5, dtype=np.int32)
print(scales)
#scales = np.arange(15,600, 4)
ax = scg.cws(t, y1+y2, scales, figsize=(12,6), ylabel="Period [Seoconds]", xlabel='Seconds', yscale='log')
ticks = ax.set_yticks([2,4,8, 16,32])
ticks = ax.set_yticklabels([2,4,8, 16,32])
print('Done')
'''
plt.show(block = True)
