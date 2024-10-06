import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


sampling_rate = 100
duration = 1000
time = np.arange(0, duration, 1/sampling_rate)
n_samples = len(time)

noise = 0.05 * np.random.normal(size=n_samples)

def generate_seismic_event(start_time, duration, amplitude):
    event = np.zeros_like(time)
    start_idx = int(start_time * sampling_rate)
    end_idx = int((start_time + duration) * sampling_rate)
    event[start_idx:end_idx] = amplitude * np.sin(np.linspace(0, np.pi, end_idx - start_idx))
    return event

seismic_events = np.zeros_like(time)
events = [
    (100, 10, 1.0),   # ssm de 10s -> 100s amplitude 1.0
    (300, 5, 0.8),    # ssm de 5s -> 300s amplitude 0.8
    (700, 15, 1.2),   # ssm de 15s -> 700s amplitude 1.2
]
for start, duration, amplitude in events:
    seismic_events += generate_seismic_event(start, duration, amplitude)

signal = noise + seismic_events
plt.figure(figsize=(10, 4))
plt.plot(time, signal, label="Signal Sismique Simulé")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.title("Signal Sismique avec Bruit et Événements Aléatoires")
plt.legend()
plt.show()
df = pd.DataFrame({'time': time, 'amplitude': signal})
df.to_csv('synthetic_seismic_data.csv', index=False)

print("Données simulées générées et sauvegardées sous 'synthetic_seismic_data.csv'")
