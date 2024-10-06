import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_seismic_event(file_path):
    # Charger les données
    df = pd.read_csv(file_path)
    df.columns = ['time_abs', 'time_rel_sec', 'velocity_m_s']

    # Préparer les données // Data cleaning
    df['time_rel_sec'] = pd.to_numeric(df['time_rel_sec'], errors='coerce')
    df['velocity_m_s'] = pd.to_numeric(df['velocity_m_s'], errors='coerce')

    # Calculer la fréquence principale
    frequencies = np.fft.fftfreq(len(df['velocity_m_s']), d=np.mean(np.diff(df['time_rel_sec'])))
    main_frequency = np.abs(frequencies[np.argmax(np.abs(np.fft.fft(df['velocity_m_s'])))])
    df['power'] = (df['velocity_m_s'] ** 2) / np.sqrt(main_frequency)

    # Calcul de l'énergie
    df['energy'] = np.cumsum(df['power'] * np.diff(np.concatenate(([0], df['time_rel_sec']))))

    # Identifier l'oscillation la plus élevée
    max_power_index = df['power'].idxmax()
    max_power_value = df['power'].max()

    # Créer une figure pour la puissance et l'énergie
    fig, ax1 = plt.subplots(figsize=(14, 8))  # Taille de la figure plus grande

    # Tracer la puissance
    color = 'tab:blue'
    ax1.set_xlabel('Temps Relatif (sec)', fontsize=12)
    ax1.set_ylabel('Puissance (W)', color=color, fontsize=12)
    ax1.plot(df['time_rel_sec'], df['power'], label='Puissance', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Énergie (Joules)', color=color, fontsize=12)
    ax2.plot(df['time_rel_sec'], df['energy'], label='Énergie', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Ajouter un trait pour l'oscillation la plus élevée
    ax1.axvline(x=df['time_rel_sec'].iloc[max_power_index], color='green', linestyle='--',
                label='Début de lévénement (Oscillation Max)')
    plt.text(df['time_rel_sec'].iloc[max_power_index], max_power_value + 0.01, 'Début', color='green', fontsize=12)

    # Identifier l'oscillation la plus basse après l'oscillation la plus élevée
    min_after_max_index = df['power'][max_power_index:].idxmin() + max_power_index
    min_after_max_value = df['power'].min()

    # Utiliser iloc pour éviter le KeyError
    if min_after_max_index < len(df):
        ax1.axvline(x=df['time_rel_sec'].iloc[min_after_max_index], color='purple', linestyle='--',
                    label='Fin de lévénement (Oscillation Min)')
        plt.text(df['time_rel_sec'].iloc[min_after_max_index], min_after_max_value + 0.01, 'Fin', color='purple',
                 fontsize=12)

    # Ajouter des titres et une grille
    plt.title('Énergie et Puissance au Cours du Temps', fontsize=14)
    ax1.grid(True)

    # Ajuster les marges
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    # Ajouter une légende
    fig.tight_layout(pad=1.0)  # Pour éviter le chevauchement
    plt.show()

file_path = 'xa.s12.00.mhz.1971-10-20HR00_evid00044.csv'
plot_seismic_event(file_path)