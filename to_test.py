import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from obspy import read
import glob

def load_data(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        df.columns = ['time_abs', 'time_rel_sec', 'velocity_m_s']
    elif file_path.endswith('.mseed'):
        st = read(file_path)
        df = pd.DataFrame({
            'time_abs': st[0].times('timestamp'),
            'time_rel_sec': st[0].times(),
            'velocity_m_s': st[0].data
        })

    df['time_rel_sec'] = pd.to_numeric(df['time_rel_sec'], errors='coerce')
    df['velocity_m_s'] = pd.to_numeric(df['velocity_m_s'], errors='coerce')
    return df


def calculate_power_and_energy(df):
    frequencies = np.fft.fftfreq(len(df['velocity_m_s']), d=np.mean(np.diff(df['time_rel_sec'])))
    main_frequency = np.abs(frequencies[np.argmax(np.abs(np.fft.fft(df['velocity_m_s'])))])

    df['power'] = (df['velocity_m_s'] ** 2) / np.sqrt(main_frequency)
    df['energy'] = np.cumsum(df['power'] * np.diff(np.concatenate(([0], df['time_rel_sec']))))

    return df


def detect_seismic_events(df, power_threshold_factor=5, energy_threshold_factor=5, min_distance=1000):
    df['smoothed_power'] = savgol_filter(df['power'], window_length=51, polyorder=3)

    power_threshold = np.mean(df['smoothed_power']) + power_threshold_factor * np.std(df['smoothed_power'])
    energy_rate = np.diff(df['energy']) / np.diff(df['time_rel_sec'])
    energy_threshold = np.mean(energy_rate) + energy_threshold_factor * np.std(energy_rate)

    power_peaks, _ = find_peaks(df['smoothed_power'], height=power_threshold, distance=min_distance)
    energy_peaks, _ = find_peaks(energy_rate, height=energy_threshold, distance=min_distance)

    all_peaks = sorted(set(power_peaks) | set(energy_peaks))

    return all_peaks


def find_event_boundaries(df, peak, window_size=500, power_threshold_factor=0.1, energy_threshold_factor=0.1):
    start_index = max(0, peak - window_size)
    end_index = min(len(df), peak + window_size)

    event_window = df.iloc[start_index:end_index]

    power_threshold = power_threshold_factor * df['smoothed_power'].iloc[peak]
    energy_threshold = energy_threshold_factor * (df['energy'].iloc[peak] - df['energy'].iloc[start_index])

    # Trouver le début de l'événement
    for i in range(peak, start_index, -1):
        if df['smoothed_power'].iloc[i] < power_threshold and (
                df['energy'].iloc[peak] - df['energy'].iloc[i]) < energy_threshold:
            start = i
            break
    else:
        start = start_index

    # Trouver la fin de l'événement
    for i in range(peak, end_index):
        if df['smoothed_power'].iloc[i] < power_threshold and (
                df['energy'].iloc[i] - df['energy'].iloc[peak]) < energy_threshold:
            end = i
            break
    else:
        end = end_index

    return start, end


def plot_results(df, events):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 20), sharex=True)

    ax1.plot(df['time_rel_sec'], df['velocity_m_s'], label='Vitesse', color='green')
    ax1.set_ylabel('Vitesse (m/s)')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(df['time_rel_sec'], df['power'], label='Puissance', color='blue', alpha=0.5)
    ax2.plot(df['time_rel_sec'], df['smoothed_power'], label='Puissance lissée', color='darkblue')
    ax2.set_ylabel('Puissance (W)')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True)

    ax3.plot(df['time_rel_sec'], df['energy'], label='Énergie cumulée', color='red')
    ax3.set_xlabel('Temps Relatif (sec)')
    ax3.set_ylabel('Énergie (J)')
    ax3.legend()
    ax3.grid(True)

    for start, peak, end in events:
        for ax in (ax1, ax2, ax3):
            ax.axvline(x=df['time_rel_sec'].iloc[start], color='green', linestyle='--', alpha=0.7)
            ax.axvline(x=df['time_rel_sec'].iloc[peak], color='purple', linestyle='--', alpha=0.7)
            ax.axvline(x=df['time_rel_sec'].iloc[end], color='red', linestyle='--', alpha=0.7)

    plt.title('Analyse des Données Sismiques')
    plt.tight_layout()
    plt.show()


def process_file(file_path):
    print(f"Traitement du fichier : {file_path}")
    df = load_data(file_path)
    df = calculate_power_and_energy(df)
    peaks = detect_seismic_events(df)

    events = []
    for peak in peaks:
        start, end = find_event_boundaries(df, peak)
        events.append((start, peak, end))

    plot_results(df, events)

    return {
        'file': file_path,
        'events': [(df['time_abs'].iloc[start], df['time_abs'].iloc[peak], df['time_abs'].iloc[end]) for
                   start, peak, end in events]
    }


def process_all_files(directory):
    file_patterns = ['*.csv', '*.mseed']
    all_files = []
    for pattern in file_patterns:
        all_files.extend(glob.glob(f"{directory}/{pattern}"))

    results = [process_file(file) for file in all_files]
    return results


# Exemple d'utilisation
directory = '.'
results = process_all_files(directory)

# Afficher les résultats
for result in results:
    print(f"\nFichier: {result['file']}")
    print("Événements détectés (début, pic, fin):")
    for start, peak, end in result['events']:
        print(f"  Début: {start}")
        print(f"  Pic: {peak}")
        print(f"  Fin: {end}")
        print()