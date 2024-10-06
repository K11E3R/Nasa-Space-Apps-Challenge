import pandas as pd
import glob
import os
import numpy as np

file_pattern = os.path.join(".", '*.csv')

all_files = glob.glob(file_pattern)

def puissance_calculator():

    for i in all_files:
        df = pd.read_csv(i, delimiter=',')
        if 'frequency(Hz)' in df.columns :
            df['puissance'] = (df['velocity(m/s)'] ** 2) / np.sqrt(df['frequency(Hz)'])
            output_file = os.path.join(".", f"modified_Puissance_added_{os.path.basename(i)}")
            df.to_csv(output_file, index=False)
            print(f"File {i} processed and saved as {output_file}")



def add_base():
    for file in all_files:
        df = pd.read_csv(file, delimiter=',')

        if 'time_abs(%Y-%m-%dT%H:%M:%S.%f)' in df.columns:
            df['time_abs'] = pd.to_datetime(df['time_abs(%Y-%m-%dT%H:%M:%S.%f)'])
            df['time_diff'] = df['time_abs'].diff().dt.total_seconds()
            df['frequency(Hz)'] = 1 / df['time_diff']
            df['frequency(Hz)'] = df['frequency(Hz)'].bfill()

            output_file = os.path.join(".", f"modified_{os.path.basename(file)}")
            df.to_csv(output_file, index=False)

            print(f"File {file} processed and saved as {output_file}")
        else:
            print(f"Time column not found in {file}")
            continue

if __name__=='__main__':
    puissance_calculator()