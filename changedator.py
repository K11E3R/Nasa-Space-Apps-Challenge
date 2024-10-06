import numpy as np
from scipy.fft import fft, fftfreq


def calculate_frequency(df, sampling_rate):
    N = len(df['velocity'])
    fft_values = fft(df['velocity'])
    frequencies = fftfreq(N, d=1 / sampling_rate)
    positive_frequencies = frequencies[:N // 2]
    fft_magnitudes = np.abs(fft_values[:N // 2])

    return positive_frequencies, fft_magnitudes


sampling_rate = 1
for df in df_list:
    frequencies, magnitudes = calculate_frequency(df, sampling_rate)
    df['frequency'] = magnitudes  # Ajouter les fréquences au DataFrame
