import pandas as pd
import matplotlib.pyplot as plt

dt = pd.read_csv("modified_Puissance_added_modified_xa.s12.00.mhz.1970-04-26HR00_evid00007.csv")
dt.columns = dt.columns.str.strip()



# Plot raw da
# ta to inspect trends
plt.figure(figsize=(12, 6))
plt.plot(dt['frequency(Hz)'], label='Frequency (Hz)', color='blue', alpha=0.5)
plt.plot(dt['puissance'], label='Puissance', color='orange', alpha=0.5)
plt.title('Raw Data: Frequency and Puissance')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))
plt.scatter(dt['frequency(Hz)'], dt['puissance'], color='blue', alpha=0.5)
plt.yscale('log')  # Logarithmic scale for better visibility
plt.title('Puissance vs. Frequency (Log Scale)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Puissance')
plt.grid()
plt.show()
