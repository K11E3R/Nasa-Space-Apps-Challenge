import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from obspy.signal.invsim import cosine_taper
from obspy.signal.filter import bandpass
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
from obspy.imaging.spectrogram import spectrogram
import seaborn as sns

# TODO :


def stochastic_source_model(magnitude, corner_frequency, stress_drop, sampling_rate, duration): # DONE docs
    time = np.arange(0, duration, 1 / sampling_rate)
    omega = 2 * np.pi * np.fft.rfftfreq(len(time), d=1 / sampling_rate)
    omega0 = 2 * np.pi * corner_frequency

    # compite source spectrum
    source_spectrum = np.zeros_like(omega)
    idx = omega != 0
    source_spectrum[idx] = stress_drop * 1e6 / (1 + (omega[idx] / omega0) ** 2)
    source_spectrum[0] = 0

    # random phase && Combine amplitude and phase && Inverse FFT to get time domain signal
    phase = np.random.uniform(0, 2 * np.pi, len(omega))
    spectrum = np.sqrt(source_spectrum) * np.exp(1j * phase)
    source_time_function = np.fft.irfft(spectrum, n=len(time))

    # magnitude scaling
    source_time_function *= 10 ** (1.5 * magnitude - 16.1)

    return source_time_function


def path_effects(signal, distance, sampling_rate, q_factor=1000): # DONE docs
    time = np.arange(len(signal)) / sampling_rate
    omega = 2 * np.pi * np.fft.rfftfreq(len(time), d=1 / sampling_rate)

    # Geometrical spreading
    signal /= np.sqrt(distance)

    # Anelastic attenuation
    spectrum = np.fft.rfft(signal)
    spectrum *= np.exp(-np.pi * distance * omega / (q_factor * 3500))  # Assuming average velocity of 3.5 km/s

    return np.fft.irfft(spectrum, n=len(signal))


def site_response(signal, sampling_rate, resonance_freq=5, amplification=2): # DONE docs
    freq = np.fft.rfftfreq(len(signal), d=1 / sampling_rate)
    response = 1 + (amplification - 1) / (1 + ((freq - resonance_freq) / (0.5 * resonance_freq)) ** 2)
    spectrum = np.fft.rfft(signal) * response
    return np.fft.irfft(spectrum, n=len(signal))


def generate_noise(duration, sampling_rate): # DONE docs
    n_samples = int(duration * sampling_rate)
    freq = np.fft.rfftfreq(n_samples, d=1 / sampling_rate)

    # high Noise Model (Peterson, 1993)
    # small constant avoid division by zero (limit -> ( -inf ))
    epsilon = 1e-6
    noise_model = 10 ** (-2.5) * (freq + epsilon) ** (-0.5)

    # flattern the spectrum for very low frequencies
    noise_model[freq < 0.1] = noise_model[np.argmin(np.abs(freq - 0.1))]

    # randomness
    noise_model *= np.exp(np.random.normal(0, 0.5, len(freq)))
    phase = np.random.uniform(0, 2 * np.pi, len(freq))
    spectrum = np.sqrt(noise_model) * np.exp(1j * phase)

    # inverse FFT to get time domain signal
    noise = np.fft.irfft(spectrum, n=n_samples)

    return noise


def generate_seismic_event(time, event_time, magnitude, distance, sampling_rate): # TO CHECK
    duration = len(time) / sampling_rate
    stress_drop = 10 ** np.random.uniform(0, 2)  # 1-100 bars
    corner_frequency = 4.9e6 * 0.37 * (stress_drop / 1e6) ** (1 / 3) * 10 ** (-0.5 * magnitude)
    source = stochastic_source_model(magnitude, corner_frequency, stress_drop, sampling_rate, duration)
    signal = path_effects(source, distance, sampling_rate)
    signal = site_response(signal, sampling_rate)
    arrival_time = int((event_time + distance / 6) * sampling_rate)  # P-wave velocity -- 6 km/s
    padded_signal = np.zeros_like(time)
    padded_signal[arrival_time:arrival_time + len(signal)] = signal[:len(padded_signal) - arrival_time]

    return padded_signal

def generate_realistic_seismic_data(duration, sampling_rate, num_events):
    time = np.arange(0, duration, 1 / sampling_rate)
    signal = generate_noise(duration, sampling_rate)

    for _ in range(num_events):
        event_time = np.random.uniform(0, duration)
        magnitude = np.random.uniform(2, 6)
        distance = np.random.uniform(10, 500)  # km
        event = generate_seismic_event(time, event_time, magnitude, distance, sampling_rate)
        signal += event

    signal = bandpass(signal, 0.1, 20, sampling_rate, corners=4)
    taper = cosine_taper(len(signal), p=0.1)
    signal *= taper

    return time, signal

def plot_seismogram(time, data, title, filename):
    plt.figure(figsize=(20, 8))
    plt.plot(time, data, linewidth=0.5)
    plt.title(title)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# to recheck
def plot_spectrogram(data, sampling_rate, title, filename):
    fig = plt.figure(figsize=(20, 8))
    spec_fig = spectrogram(data, sampling_rate, per_lap=0.9, wlen=2, dbscale=True,
                           title=title, show=False)
    fig.colorbar(spec_fig[0], label='Amplitude (dB)')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_psd(data, sampling_rate, title, filename):
    f, Pxx = signal.welch(data, fs=sampling_rate, nperseg=8192)
    plt.figure(figsize=(20, 8))
    plt.semilogy(f, Pxx)
    plt.title(title)
    plt.xlabel('frequency (hz)')
    plt.ylabel('power spectral density')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def detect_events(data, sampling_rate):
    sta_window = 1  # short-term average window in seconds
    lta_window = 30  # long-term average window in seconds
    trigger_ratio = 3  # STA/LTA ratio to trigger event detection
    detrigger_ratio = 1  # STA/LTA ratio to end event detection

    sta_samples = int(sta_window * sampling_rate)
    lta_samples = int(lta_window * sampling_rate)

    cft = recursive_sta_lta(data, sta_samples, lta_samples)
    triggers = trigger_onset(cft, trigger_ratio, detrigger_ratio)

    return triggers


def plot_events(time, data, triggers, title, filename):
    plt.figure(figsize=(20, 8))
    plt.plot(time, data, linewidth=0.5)
    for on, off in triggers:
        plt.axvspan(time[on], time[off], color='red', alpha=0.3)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_event_spectrograms(time, data, triggers, sampling_rate, filename_prefix):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Spectrogram for the entire signal
    ax1.set_title('Spectrogramme complet')
    ax1.specgram(data, NFFT=256, Fs=sampling_rate, noverlap=128)
    ax1.set_xlabel('Temps (s)')
    ax1.set_ylabel('Fréquence (Hz)')

    # Loop through all trigger events
    for event_index in range(len(triggers)):
        event_start = triggers[event_index] - 5  # 5 seconds before the trigger
        event_end = triggers[event_index] + 10    # 10 seconds after the trigger

        # Convert to integers to index into the array
        event_start_index = int(event_start * sampling_rate)
        event_end_index = int(event_end * sampling_rate)

        # Ensure indices are within bounds
        event_start_index = max(0, event_start_index)  # Avoid negative index
        event_end_index = min(len(data), event_end_index)  # Avoid going out of bounds

        # Extract the event data
        event_data = data[event_start_index:event_end_index]

        # Calculate the spectrogram
        spec_fig, spec_data, freq, t = spectrogram(event_data, sampling_rate, per_lap=0.9, wlen=2, dbscale=True, show=False)

        # Plot the spectrogram for the current event
        ax2.set_title(f'Spectrogramme de l\'événement détecté {event_index + 1}')
        ax2.imshow(spec_fig, aspect='auto', origin='lower', extent=[t[0], t[-1], freq[0], freq[-1]])
        ax2.set_xlabel('Temps (s)')
        ax2.set_ylabel('Fréquence (Hz)')

    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_spectrogram.png')
    plt.show()

def analyze_seismic_data(time, data, sampling_rate):
    plot_seismogram(time, data, 'Full Synthetic Seismogram', 'full_seismogram.png')

    plt.figure(figsize=(20, 8))
    plt.specgram(data, NFFT=256, Fs=sampling_rate, noverlap=128)
    plt.title('Full Data Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    plt.savefig('full_spectrogram.png', dpi=300, bbox_inches='tight')
    plt.close()

    plot_psd(data, sampling_rate, 'Power Spectral Density', 'power_spectral_density.png')

    triggers = detect_events(data, sampling_rate)
    plot_events(time, data, triggers, 'Detected Seismic Events', 'detected_events.png')
    plot_event_spectrograms(time, data, triggers, sampling_rate, 'event_analysis')

    plt.figure(figsize=(20, 8))
    sns.histplot(data, kde=True)
    plt.title('Amplitude Distribution')
    plt.xlabel('Amplitude')
    plt.ylabel('Count')
    plt.savefig('amplitude_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    energy = np.cumsum(data ** 2)
    plt.figure(figsize=(20, 8))
    plt.plot(time, energy)
    plt.title('Cumulative Energy Release')
    plt.xlabel('Time (s)')
    plt.ylabel('Cumulative Energy')
    plt.savefig('cumulative_energy.png', dpi=300, bbox_inches='tight')
    plt.close()

print("Generating synthetic seismic data...")
duration = 3600 * 24  #24h
sampling_rate = 100  #hz
num_events = 50

time, seismic_signal = generate_realistic_seismic_data(duration, sampling_rate, num_events)

df = pd.DataFrame({'time': time, 'amplitude': seismic_signal})
df.to_csv('highly_realistic_seismic_data.csv', index=False)
print("Highly realistic seismic data generated and saved to 'highly_realistic_seismic_data.csv'")
print("Analyzing seismic data...")
analyze_seismic_data(time, seismic_signal, sampling_rate) # DONE docs (CHECKED)
print("Analysis complete. Plots have been saved.")
print("\nSummary Statistics:")
print(f"Duration: {duration} seconds ({duration / 3600} hours)")
print(f"Sampling Rate: {sampling_rate} Hz")
print(f"Number of Samples: {len(seismic_signal)}")
print(f"Number of Simulated Events: {num_events}")
print(f"Mean Amplitude: {np.mean(seismic_signal):.6f}")
print(f"Standard Deviation: {np.std(seismic_signal):.6f}")
print(f"Maximum Amplitude: {np.max(np.abs(seismic_signal)):.6f}")
triggers = detect_events(seismic_signal, sampling_rate) # ROTATION TO CHECK (CORRELATION)
event_durations = [(off - on) / sampling_rate for on, off in triggers]
print(f"\nDetected Events: {len(triggers)}")
print(f"Mean Event Duration: {np.mean(event_durations):.2f} seconds")
print(f"Median Event Duration: {np.median(event_durations):.2f} seconds")
print(f"Minimum Event Duration: {np.min(event_durations):.2f} seconds")
print(f"Maximum Event Duration: {np.max(event_durations):.2f} seconds")