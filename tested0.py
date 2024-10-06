import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from obspy import read
import glob


def load_data(file_path):
    try:
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
    except Exception as e:
        print(f"Erreur lors du chargement du fichier {file_path}: {str(e)}")
        return None


def calculate_power_and_energy(df):
    try:
        frequencies = np.fft.fftfreq(len(df['velocity_m_s']), d=np.mean(np.diff(df['time_rel_sec'])))
        main_frequency = np.abs(frequencies[np.argmax(np.abs(np.fft.fft(df['velocity_m_s'])))])

        df['power'] = (df['velocity_m_s'] ** 2) / np.sqrt(main_frequency)
        df['energy'] = np.cumsum(df['power'] * np.diff(np.concatenate(([0], df['time_rel_sec']))))
        df['smoothed_power'] = savgol_filter(df['power'], window_length=51, polyorder=3)

        return df
    except Exception as e:
        print(f"Erreur lors du calcul de la puissance et de l'énergie: {str(e)}")
        return None


def find_main_oscillation(df, power_threshold_factor=5, min_duration=10, max_duration=300):
    try:
        if len(df) == 0:
            print("Le DataFrame est vide.")
            return None, None

        power_threshold = np.mean(df['smoothed_power']) + power_threshold_factor * np.std(df['smoothed_power'])

        above_threshold = df['smoothed_power'] > power_threshold
        oscillation_starts = np.where(above_threshold[:-1] == False)[0]
        oscillation_ends = np.where(above_threshold[1:] == False)[0] + 1

        min_length = min(len(oscillation_starts), len(oscillation_ends))
        oscillation_starts = oscillation_starts[:min_length]
        oscillation_ends = oscillation_ends[:min_length]

        valid_oscillations = []
        for start, end in zip(oscillation_starts, oscillation_ends):
            if start >= len(df) or end >= len(df):
                continue
            duration = df['time_rel_sec'].iloc[end] - df['time_rel_sec'].iloc[start]
            if min_duration <= duration <= max_duration:
                amplitude = df['smoothed_power'].iloc[start:end].max()
                valid_oscillations.append((start, end, amplitude, duration))

        if not valid_oscillations:
            print("Aucune oscillation valide trouvée.")
            return None, None

        # Sélectionner l'oscillation avec la plus grande amplitude
        main_oscillation = max(valid_oscillations, key=lambda x: x[2])

        return main_oscillation[0], main_oscillation[1]  # Retourner start et end
    except Exception as e:
        print(f"Erreur lors de la recherche de l'oscillation principale: {str(e)}")
        return None, None


def plot_main_oscillation(df, start, end):
    try:
        if start is None or end is None or start >= len(df) or end >= len(df) or start < 0 or end < 0:
            print("Indices de début ou de fin invalides pour le tracé.")
            return

        start, end = min(start, end), max(start, end)
        end = min(end, len(df) - 1)
        event_df = df.iloc[start:end + 1]

        if len(event_df) == 0:
            print("Aucune donnée à tracer pour l'événement.")
            return

        peak = start + event_df['smoothed_power'].idxmax() - event_df.index[0]
        peak = max(0, min(peak, len(event_df) - 1))

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

        ax1.plot(event_df['time_rel_sec'], event_df['velocity_m_s'], label='Vitesse', color='green')
        ax1.set_ylabel('Vitesse (m/s)')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(event_df['time_rel_sec'], event_df['power'], label='Puissance', color='blue', alpha=0.5)
        ax2.plot(event_df['time_rel_sec'], event_df['smoothed_power'], label='Puissance lissée', color='darkblue')
        ax2.set_ylabel('Puissance (W)')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True)

        ax3.plot(event_df['time_rel_sec'], event_df['energy'], label='Énergie cumulée', color='red')
        ax3.set_xlabel('Temps Relatif (sec)')
        ax3.set_ylabel('Énergie (J)')
        ax3.legend()
        ax3.grid(True)

        for ax in (ax1, ax2, ax3):
            ax.axvline(x=event_df['time_rel_sec'].iloc[0], color='green', linestyle='--', alpha=0.7, label='Début')
            ax.axvline(x=event_df['time_rel_sec'].iloc[peak], color='purple', linestyle='--', alpha=0.7, label='Pic')
            ax.axvline(x=event_df['time_rel_sec'].iloc[-1], color='red', linestyle='--', alpha=0.7, label='Fin')

        duration = event_df['time_rel_sec'].iloc[-1] - event_df['time_rel_sec'].iloc[0]
        plt.title(f'Oscillation Principale (Durée: {duration:.2f}s)')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Erreur lors de la création du graphique: {str(e)}")
        print(f"Détails - start: {start}, end: {end}, len(df): {len(df)}")


def process_file(file_path):
    print(f"Traitement du fichier : {file_path}")
    df = load_data(file_path)
    if df is None or len(df) == 0:
        print(f"Impossible de charger ou DataFrame vide pour {file_path}")
        return {'file': file_path, 'event': None}

    df = calculate_power_and_energy(df)
    if df is None:
        print(f"Erreur lors du calcul de la puissance et de l'énergie pour {file_path}")
        return {'file': file_path, 'event': None}

    start, end = find_main_oscillation(df)
    if start is not None and end is not None:
        plot_main_oscillation(df, start, end)
        return {
            'file': file_path,
            'event': (df['time_abs'].iloc[start], df['time_abs'].iloc[end])
        }
    else:
        print(f"Aucune oscillation principale détectée pour {file_path}")
        return {'file': file_path, 'event': None}


def process_all_files(directory):
    file_patterns = ['*.csv', '*.mseed']
    all_files = []
    for pattern in file_patterns:
        all_files.extend(glob.glob(f"xa.s12.00.mhz.1971-10-20HR00_evid00044.csv"))

    results = []
    for file in all_files:
        try:
            result = process_file(file)
            results.append(result)
        except Exception as e:
            print(f"Erreur lors du traitement du fichier {file}: {str(e)}")

    return results


# Exemple d'utilisation
if __name__ == "__main__":
    directory = '.'
    results = process_all_files(directory)

    # Afficher les résultats
    for result in results:
        print(f"\nFichier: {result['file']}")
        if result['event']:
            start, end = result['event']
            print(f"Événement sismique principal détecté:")
            print(f"  Début: {start}")
            print(f"  Fin: {end}")
        else:
            print("Aucun événement sismique principal détecté.")