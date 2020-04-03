import matplotlib.pyplot as plt
import numpy as np
import random
import cmath
import math
import time

n = 12
omega = 1100
N = 256

def create_signals(n, N, W):
    generated_signal = np.zeros(N)
    for i in range(n):
        fi = 2 * math.pi * random.random()
        A = 5 * random.random()
        w = W - i * W / (n)
        x = A * np.sin(np.arange(0, N, 1) * w + fi)
        generated_signal += x

    return generated_signal


def create_plot(arr, x_label, y_label, title, legend, file_name=None):
    result, = plt.plot(range(len(arr)), arr, '-', label=legend)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    return result



def dft(signal):
    start = time.time()
    N = len(signal)
    spectre = np.zeros(N, dtype=np.complex64)
    for p in range(N):
        spectre[p] = np.dot(signal, np.cos(2 * math.pi * p / N * np.linspace(0, N-1, N))) \
        -1j * np.dot(signal, np.sin(2 * math.pi * p / N * np.linspace(0, N-1, N)))
    print(f'Execution time DFT: {time.time() - start}')
    return spectre



def fft(signal):
    start = time.time()
    N = len(signal)
    spectre = np.zeros(N, dtype=np.complex64)
    for p in range(N // 2):
        E_m = np.dot(signal[0:N:2], np.cos(2 * math.pi * p / (N / 2) * np.arange(0, N // 2, 1))) - 1j * np.dot(signal[0:N:2],
              np.sin(2 * math.pi * p / (N / 2) * np.arange(0, N // 2, 1)))
        W_p = (np.cos(2 * math.pi * p / N) - 1j * np.sin(2 * math.pi * p / N))
        O_m = np.dot(signal[1:N:2], np.cos(2 * math.pi * p / (N / 2) * np.arange(0, N // 2, 1))) - 1j * np.dot(signal[1:N:2],
              np.sin(2 * math.pi * p / (N / 2) * np.arange(0, N // 2, 1)))
        spectre[p] = E_m + W_p * O_m
        spectre[p + N // 2] = E_m - W_p * O_m
    print(f'Execution time FFT: {time.time() - start}')
    return spectre

signal = create_signals(n, N, omega)
plot = create_plot(signal, "t", "x(t)", "Signal", "X(t)", "blue")
plt.grid()
plt.savefig('signal.png')


spectr_dft = dft(signal)
polar_spectr_dft = np.array(list(map(lambda x: cmath.polar(x), spectr_dft)))[:, 0]
ampl_dft = create_plot(polar_spectr_dft, "p", "A(p)", "Polar Spectr DFT", "Amplitude")
plt.legend(handles=[ampl_dft], loc='upper right')
plt.grid()
plt.savefig('DFT.png')



spectr_fft = fft(signal)
polar_spectr_fft = np.array(list(map(lambda x: cmath.polar(x), spectr_fft)))[:, 0]
ampl_fft = create_plot(polar_spectr_fft, "p", "A(p)", "Polar Spectr FFT", "Amplitude")
plt.legend(handles=[ampl_fft], loc='upper right')
plt.grid()
plt.savefig('FFT.png')
