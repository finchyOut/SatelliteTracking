from orbits import Satellite, Sun, Moon
from orbitPlots import plot_paths_xy, plot_paths_3d, plot2d, plot3d, EARTH_RADIUS_KM
#how to call: 
#/opt/anaconda3/envs/cleanorbit/bin/python "/Users/emmafinch/Desktop/Thesis/Python Code/simModel/mainRun.py"

#Create bodies with perfect circular orbits
#the phase just determines where it starts in its loop
sun  = Sun(phase_deg=0.0)   
moon = Moon(phase_deg=45.0)   

# Optionally propagate Sun/Moon on their own timelines (for plotting/logging)
# These do NOT affect their dynamics; Satellite will query position_at(t).
#so this is just if you personally want to make a separate log to see where
#the sun and moon are positioned
#sun.propagate(duration=24*3600)    # simulate 1 day of Sun positions (optional)
#moon.propagate(duration=24*3600)   # simulate 1 day of Moon positions (optional)

# Satellite with third-body effects enabled
sat = Satellite(
    sat_id='01',
    longitude_deg=45,
    noise_std=0.08,
    max_drift_km=1000,
    time_step=10,
    bodies=[sun, moon]   # <-- enable Sun+Moon gravity
)


sat.propagate(duration=86400 ) #propagates for one day aka one loop.
df = sat.get_log()
df.to_csv("simModel/simLog.csv", index=False)

# 1) Nice zoom into Earth neighborhood (recommended)
plot_paths_xy(df, focus="geo")
plot_paths_3d(df, focus="geo")

# 2) If you want the full system including the Sun (everything in one frame)
plot_paths_xy(df, focus="all")
plot_paths_3d(df, focus="all")

# 3) just plots the satellite around earth station keeping vs not station keeping
plot2d(df)
plot3d(df)