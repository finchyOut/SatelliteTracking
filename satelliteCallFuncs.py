from pathlib import Path
from bangBangSat import SatelliteBB
from pdSat import SatellitePD
from lqrSat import SatelliteLQR
from sunAndMoonOrbits import SkyfieldBody, Sun, Moon
from satellitePlots import plot3d, plot2d, plot_accel_x, plot_accel_y, plot_accel_z, plot_error, plot_paths_3d, plot_paths_xy
import pandas as pd
#These are the helper functions for easier calls to create different satellite runs


# The function call for the Bang Bang controller
def callBB( daysToProp, time_Step, longitudeDeg, maxDrift, noise_std, maxA, satId):
    days = daysToProp
    totalProp = 86400 * days # 1 day times X daysin seconds
    timeStep = time_Step #in secods

    #Create bodies with perfect circular orbits
    sun  = Sun(phase_deg= 0.0, time_step= timeStep, duration = totalProp)
    moon = Moon(phase_deg = 45.0, time_step= timeStep, duration = totalProp)
    sun.propagate(totalProp)
    moon.propagate(totalProp)

    # Satellite with third-body effects enabled
    sat = SatelliteBB(
        sat_id= satId,
        longitude_deg=longitudeDeg,
        max_drift_km=maxDrift,
        time_step=timeStep, #somewhere 20-70
        noise_std= noise_std * .5* timeStep**2,
        bodies=[sun, moon],
        a_max = maxA #This is a constraint on the magnitude of the total accel vec
    )


    sat.propagate(duration=totalProp ) #propagates for one day aka one loop.
    df = sat.get_log()
    df.to_csv("setPolicies/savedLogs/bangBangSatSim.csv", index=False)

    return df

# The function call for the Pd Controller
def callPd(daysToProp, time_Step, longitudeDeg, maxDrift, noise_std, maxA, satId):
    days = daysToProp
    totalProp = 86400 * days # 1 day times X daysin seconds
    timeStep = time_Step #in secods

    #Create bodies with perfect circular orbits
    sun  = Sun(phase_deg= 0.0, time_step= timeStep, duration = totalProp)
    moon = Moon(phase_deg = 45.0, time_step= timeStep, duration = totalProp)
    sun.propagate(totalProp)
    moon.propagate(totalProp)

    sat = SatellitePD(
    sat_id= satId,
    longitude_deg= longitudeDeg,
    max_drift_km= maxDrift,
    time_step= time_Step, #somewhere 20-70
    noise_std= noise_std * .5* timeStep**2,
    bodies=[sun, moon],   # <-- enable Sun+Moon gravity
)

    sat.propagate(duration=totalProp) #propagates for one day aka one loop.
    df = sat.get_log()
    df.to_csv("setPolicies/savedLogs/pdSatSim.csv", index=False)

    #max a is useless for simplicity tho I keep it in the func call to mirror all otherfuncs
    maxA += 1
    return df

# The function call for the LQR controller
def callLqr(daysToProp, time_Step, longitudeDeg, maxDrift, noise_std, maxA, satId):
    days = daysToProp
    totalProp = 86400 * days # 1 day times X daysin seconds
    timeStep = time_Step #in secods

    #Create bodies with perfect circular orbits
    sun  = Sun(phase_deg= 0.0, time_step= timeStep, duration = totalProp)
    moon = Moon(phase_deg = 45.0, time_step= timeStep, duration = totalProp)
    sun.propagate(totalProp)
    moon.propagate(totalProp)

    sat = SatelliteLQR(
    sat_id= satId,
    longitude_deg=longitudeDeg,
    max_drift_km=maxDrift,
    time_step=timeStep, #somewhere 20-70
    noise_std= noise_std* .5* timeStep**2,
    bodies=[sun, moon],   # <-- enable Sun+Moon gravity
)

    sat.propagate(duration=totalProp ) #propagates for one day aka one loop.
    df = sat.get_log()
    df.to_csv("setPolicies/savedLogs/lqrSatSim.csv", index=False)
    #max a is useless for simplicity tho I keep it in the func call to mirror all otherfuncs
    maxA += 1
    
    return df
    
#is one function that allows you to specify one call to the preferred control policy
def createSim(policy, daysToProp, time_Step, longitudeDeg, maxDrift, noise_std, maxA, satId):
    """Dispatch to the right simulator based on `policy` (case-insensitive)."""
    policy_norm = (policy or "").strip().lower()

    dispatch = {
        "lqr": callLqr,
        "pd":  callPd,
        "bb":  callBB,
        "bangbang": callBB,
        "bang-bang": callBB,
    }

    if policy_norm not in dispatch:
        valid = ", ".join(sorted(dispatch.keys()))
        raise ValueError(f"Unknown policy '{policy}'. Choose one of: {valid}")

    return dispatch[policy_norm](daysToProp, time_Step, longitudeDeg, maxDrift, noise_std, maxA, satId)

#get the multiple simulations from the three sims
def multipleSimsAll(numReps, daysToProp, time_Step, longitudeDeg, maxDrift, noise_std, maxA, satId):
    """
    Run numReps simulations for PD, LQR, and Bang-Bang controllers.
    Returns three lists; each element is a 1D array [x1,y1,z1, x2,y2,z2, ..., xN,yN,zN]
    for a single run.
    """
    allPd, allBb, allLqr = [], [], []
    pd_frames, lqr_frames, bb_frames = [], [], []
    cols = ['X (km)', 'Y (km)', 'Z (km)']

    # Ensure numReps is an int and iterate that many times
    for rep in range(int(numReps)):
        # Optional: give each run a unique satellite ID
        # run_sat_id = f"{satId}-{rep+1}"
        run_sat_id = satId

        # Run sims (each should return a DataFrame)
        newPd  = callPd(daysToProp, time_Step, longitudeDeg, maxDrift, noise_std, maxA, run_sat_id)
        newLqr = callLqr(daysToProp, time_Step, longitudeDeg, maxDrift, noise_std, maxA, run_sat_id)
        newBb  = callBB(daysToProp, time_Step, longitudeDeg, maxDrift, noise_std, maxA, run_sat_id)

    
        # Select only the XYZ columns and flatten to a 1D vector
        try:
            pd_vec  = newPd[cols].to_numpy().ravel(order='C')
            lqr_vec = newLqr[cols].to_numpy().ravel(order='C')
            bb_vec  = newBb[cols].to_numpy().ravel(order='C')
        except KeyError as e:
            # Helpful error if any expected column is missing
            missing = [c for c in cols if c not in newPd.columns or c not in newLqr.columns or c not in newBb.columns]
            raise KeyError(f"Missing expected columns {missing}. Got:\n"
                           f"PD:   {list(newPd.columns)}\n"
                           f"LQR:  {list(newLqr.columns)}\n"
                           f"BB:   {list(newBb.columns)}") from e

        pd_frames.append(newPd)
        lqr_frames.append(newLqr)
        bb_frames.append(newBb)

        # Accumulate
        allPd.append(pd_vec)
        allBb.append(bb_vec)
        allLqr.append(lqr_vec)
        
    # --- combine after the loop (note: concat expects a list) ---
    pd_df  = pd.concat(pd_frames,  ignore_index=True)
    lqr_df = pd.concat(lqr_frames, ignore_index=True)
    bb_df  = pd.concat(bb_frames,  ignore_index=True)

    saveAllSims(pd_df, lqr_df, bb_df, numReps)
    # Return AFTER the loop
    return allPd, allBb, allLqr

# save the multiple sims
def saveAllSims(pdSim, lqrSim, bbSim, numReps):
    """
    Save three combined CSVs for PD, LQR, and BB.
    Creates .../savedLogs/{numReps}Sims if it doesn't exist.
    """
    base_dir = Path(__file__).resolve().parent  # folder of this .py file
    folder = base_dir / "savedLogs" / f"{numReps}Sims"
    folder.mkdir(parents=True, exist_ok=True)   # <-- create the FINAL dir, not just parent

    (folder / f"pd{numReps}Sims.csv").write_text("") if pdSim.empty else pdSim.to_csv(folder / f"pd{numReps}Sims.csv", index=False)
    (folder / f"lqr{numReps}Sims.csv").write_text("") if lqrSim.empty else lqrSim.to_csv(folder / f"lqr{numReps}Sims.csv", index=False)
    (folder / f"bb{numReps}Sims.csv").write_text("") if bbSim.empty else bbSim.to_csv(folder / f"bb{numReps}Sims.csv", index=False)

def saveAllSimsTo(pdSim, bbSim, lqrSim, numReps):
    folderPath = f'setPolicies/savedLogs/{numReps}Sims'
    out = Path(folderPath)
    out.parent.mkdir(parents=True, exist_ok=True)
   
    #pd simulation
    pdTitle = f'{folderPath}/pd{numReps}Sims.csv'
    pdSim.to_csv(pdTitle, index=False)
    #bb simulation
    bbTitle = f'{folderPath}/bb{numReps}Sims.csv'
    bbSim.to_csv(bbTitle, index=False)
    #pd simulation
    lqrTitle = f'{folderPath}/lqr{numReps}Sims.csv'
    lqrSim.to_csv(lqrTitle, index=False)


"""Proof of function calls
allPd, allBb, allLqr = multipleSimsAll(numReps = 3,daysToProp= 1, time_Step = 30, longitudeDeg=45, maxDrift=36, noise_std=1e-8, maxA=1e-6, satId=1 )


print("The number of reps recorded is: ", len(allPd))
print("The starting position in allPd is", allPd[2][0:3])
print("The 2nd position in allPd is", allPd[2][3:6])
print("The 3rd position is: ", allPd[2][6:9])

# Replace with the path to your file
file_path = "setPolicies/savedLogs/pdSatSim.csv"

# Read CSV into a DataFrame
df = pd.read_csv(file_path)
dfColumns = df[['X (km)', 'Y (km)', 'Z (km)']]
dfColumns = dfColumns.head(3)
print(dfColumns)"""