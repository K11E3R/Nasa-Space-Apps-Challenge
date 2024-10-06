import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from seisbench.models import EQTransformer

# Load EQTransformer model
model = EQTransformer.from_pretrained('original')

# Model parameters
in_channels = 3
in_samples = 6000

# Load the CSV file
df = pd.read_csv('xa.s12.00.mhz.1970-01-19HR00_evid00002.csv', parse_dates=['time_abs(%Y-%m-%dT%H:%M:%S.%f)'])


# Custom Dataset for seismic data
class SeismicDataset(Dataset):
    def __init__(self, df, in_channels, in_samples):
        self.df = df
        self.in_channels = in_channels
        self.in_samples = in_samples

    def __len__(self):
        return len(self.df) - self.in_samples + 1

    def __getitem__(self, idx):
        waveform_data = self.df['velocity(m/s)'].iloc[idx:idx + self.in_samples].values

        # Normalize the data
        waveform_data = (waveform_data - np.mean(waveform_data)) / np.std(waveform_data)

        # Replicate the single channel to create 3 channels
        waveform = np.tile(waveform_data, (self.in_channels, 1))

        waveform = torch.tensor(waveform, dtype=torch.float32)
        time_abs = self.df['time_abs(%Y-%m-%dT%H:%M:%S.%f)'].iloc[idx + self.in_samples - 1]
        time_rel = self.df['time_rel(sec)'].iloc[idx + self.in_samples - 1]
        return waveform, time_abs.timestamp(), time_rel


# Custom collate function
def custom_collate(batch):
    waveforms, time_abs, time_rel = zip(*batch)
    return torch.stack(waveforms), torch.tensor(time_abs), torch.tensor(time_rel)


# Parameters
batch_size = 32

# Create dataset and dataloader
dataset = SeismicDataset(df, in_channels, in_samples)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)


# Function to detect seismic activity
def detect_seismic_activity(predicted_activity, threshold=0.3):
    activity_mask = predicted_activity > threshold
    if not activity_mask.any():
        return None, None, 0
    start_idx = activity_mask.argmax()
    end_idx = len(activity_mask) - activity_mask.flip(0).argmax() - 1
    return start_idx, end_idx, end_idx - start_idx + 1


# Evaluation
model.eval()
total_events = 0
total_samples = 0
max_detection_prob = 0
with torch.no_grad():
    for batch_idx, batch in enumerate(dataloader):
        waveforms, time_abs, time_rel = batch
        outputs = model(waveforms)

        # Interpret the model outputs
        p_prob, s_prob, detection_prob = outputs

        # Update max detection probability
        max_detection_prob = max(max_detection_prob, detection_prob.max().item())

        # Process detection probabilities
        for i in range(detection_prob.shape[0]):
            total_samples += 1
            start_idx, end_idx, duration = detect_seismic_activity(detection_prob[i])
            if start_idx is not None and end_idx is not None:
                total_events += 1
                start_time = pd.Timestamp(time_abs[i].item(), unit='s')
                end_time = start_time + pd.Timedelta(
                    seconds=(end_idx - start_idx) / 100)  # Assuming 100 Hz sampling rate
                print(f'Batch {batch_idx}, Sample {i}:')
                print(f'  Predicted start: {start_time}')
                print(f'  Predicted end: {end_time}')
                print(f'  Duration: {duration / 100:.4f} seconds')
                print(f'  Max detection probability: {detection_prob[i].max().item():.4f}')

                # Find P and S wave arrival times
                p_arrival = start_time + pd.Timedelta(seconds=p_prob[i].argmax() / 100)
                s_arrival = start_time + pd.Timedelta(seconds=s_prob[i].argmax() / 100)
                print(f'  P-wave arrival: {p_arrival}')
                print(f'  S-wave arrival: {s_arrival}')
                print()

        # Process 10 batches (adjust as needed)
        if batch_idx >= 9:
            break

print(f"Processing complete. Processed {total_samples} samples.")
print(f"Detected {total_events} potential seismic events.")
print(f"Maximum detection probability: {max_detection_prob:.4f}")

detection_probs = torch.cat([model(batch[0])[-1] for batch in dataloader])
print(f"Detection probability statistics:")
print(f"  Mean: {detection_probs.mean().item():.4f}")
print(f"  Median: {detection_probs.median().item():.4f}")
print(f"  95th percentile: {np.percentile(detection_probs.numpy(), 95):.4f}")
print(f"  99th percentile: {np.percentile(detection_probs.numpy(), 99):.4f}")