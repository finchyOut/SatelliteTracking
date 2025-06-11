
import pandas as pd
from orbit6 import Satellite

# Load satellite parameters from CSV
df = pd.read_csv("initialSatelliteMovement.csv")

# List to collect logs from all satellites
all_logs = []

# Loop over each satellite entry
for _, row in df.iterrows():

    sat_id = row['File ID']
    sma = row['Semi-major Axis (km)']
    ecc = row['Eccentricity']

    # Create Satellite object
    sat = Satellite(
        sat_id=sat_id,
        semi_major_axis=sma,
        eccentricity= 0,
        noise_freq=0.01,
        max_deviation=15,
        time_step=10
    )

    # Simulate orbit for 1 hour (3600 seconds)
    sat.propagate(duration=3600)

    # Collect logs
    log_df = sat.get_log()
    all_logs.append(log_df)

# Combine all logs
final_df = pd.concat(all_logs, ignore_index=True)

# Save to CSV
final_df.to_csv("all_satellite_logs.csv", index=False)
print("Simulation complete. Output saved to all_satellite_logs.csv")
