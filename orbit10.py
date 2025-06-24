
import pandas as pd
from orbit9 import Satellite

# Load CSV
df = pd.read_csv("initialSatelliteMovement.csv")

logs = []
for _, row in df.iterrows():
    sat = Satellite(
        sat_id=row["File ID"],
        longitude_deg=row["Longitude (deg)"],
        noise_std=0.05,
        max_drift_km=50,
        time_step=10
    )
    sat.propagate(duration=3600)
    logs.append(sat.get_log())

final_log = pd.concat(logs)
final_log.to_csv("all_satellite_logs.csv", index=False)
