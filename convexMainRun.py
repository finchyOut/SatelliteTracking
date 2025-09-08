from convexOrbitsV4 import Satellite, Sun, Moon
from convexPlots import plot_paths_xy, plot_paths_3d, plot2d, plot3d, EARTH_RADIUS_KM
#how to call: 
#/opt/anaconda3/envs/cleanorbit/bin/python "/Users/emmafinch/Desktop/Thesis/Python Code/simModel/mainRun.py"
#/opt/anaconda3/envs/cleanorbit/bin/python "/Users/emmafinch/Desktop/Thesis/Python Code/convexController /convexMainRun.py"

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
timeStep = 30
sat = Satellite(
    sat_id='01',
    longitude_deg=45,
    max_drift_km=36,
    time_step=timeStep, #somewhere 20-70
    noise_std= 1e-8 * .5* timeStep**2,
    bodies=[sun, moon],   # <-- enable Sun+Moon gravity
    gamma_km=36,   # allowed box radius around target per axis (km)
    H_steps=1,      # apply control every 10 steps (here, every 10*60 = 600 s)
    a_max=1e200
)

totalProp = 86400 *30 # 1 day times 15 days
sat.propagate(duration=totalProp ) #propagates for one day aka one loop.
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