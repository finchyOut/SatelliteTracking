# plotOrbits.py
# ====================
# This file provides plotting functions to visualize the orbits of a satellite, the Moon,
# and the Sun relative to Earth, using data from a CSV log or a pandas DataFrame.
# It includes both 2D (XY projection) and 3D plot functions, with options to zoom in on the Earth region
# or include the Sun's full orbit.

# ---- Imports ----
import numpy as np                 # Numerical operations and array handling
import pandas as pd                # For reading and handling CSV / DataFrames
import matplotlib.pyplot as plt    # Main plotting library
from typing import Union            # Type hints for functions (str or DataFrame)
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting in matplotlib (import has side-effect)

# ---- Constants ----
EARTH_RADIUS_KM = 6378.1363  # Mean radius of Earth in kilometers (used for scaling and Earth circle/sphere)

# ---------- Helper Functions ----------
def _as_df(df_or_path: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Ensures we always work with a pandas DataFrame.
    - If the input is already a DataFrame, return it unchanged.
    - If it's a string, treat it as a CSV file path and read it.
    """
    if isinstance(df_or_path, pd.DataFrame):
        return df_or_path
    return pd.read_csv(df_or_path)

def _has_cols(df: pd.DataFrame, cols):
    """
    Checks whether all specified columns exist in the DataFrame.
    Returns True if all are present, False otherwise.
    """
    return all(c in df.columns for c in cols)

def _get_cols(df: pd.DataFrame, prefix: str | None):
    """
    Returns a tuple of column names for X, Y, Z coordinates for a given object.
    - If prefix is None → assume Satellite columns: 'X (km)', 'Y (km)', 'Z (km)'.
    - If prefix is provided (e.g., "Sun" or "Moon"), prepend it to each coordinate name.
    Returns None if those columns do not exist in the DataFrame.
    """
    if prefix is None:
        cand = ("X (km)", "Y (km)", "Z (km)")
    else:
        cand = (f"{prefix} X (km)", f"{prefix} Y (km)", f"{prefix} Z (km)")
    return cand if _has_cols(df, cand) else None

def _set_equal_3d(ax, xs, ys, zs, pad_frac=0.08):
    """
    Ensures 3D plot has equal scaling for X, Y, Z axes so spheres look spherical.
    pad_frac: fraction of the max range to pad on each axis for spacing.
    """
    # Min/max for each axis
    x_min, x_max = np.nanmin(xs), np.nanmax(xs)
    y_min, y_max = np.nanmin(ys), np.nanmax(ys)
    z_min, z_max = np.nanmin(zs), np.nanmax(zs)
    # Midpoints
    cx = 0.5*(x_min + x_max)
    cy = 0.5*(y_min + y_max)
    cz = 0.5*(z_min + z_max)
    # Max dimension length
    r = max(x_max - x_min, y_max - y_min, z_max - z_min)
    r = max(r, 1.0) * (1 + pad_frac) * 0.5
    # Set symmetric limits for all axes
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(cz - r, cz + r)

# ---------- Satellite-only plots ----------
def plot3d(log_df):
    """
    Plots only the satellite's trajectory in 3D, color-coded by stationkeeping status.
    """
    # Extract position and status arrays
    x = log_df["X (km)"].values
    y = log_df["Y (km)"].values
    z = log_df["Z (km)"].values
    status = log_df["Status"].values

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot: stationkeeping vs non-stationkeeping
    ax.scatter(x[status == "stationkeeping"], y[status == "stationkeeping"], z[status == "stationkeeping"], 
               label="stationkeeping", s=3)
    ax.scatter(x[status == "non-stationkeeping"], y[status == "non-stationkeeping"], z[status == "non-stationkeeping"], 
               label="non-stationkeeping", s=3, marker='x')
    
    # Plot Earth as a sphere
    R = 6371  # km
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]  # sphere mesh
    ex = R * np.cos(u) * np.sin(v)
    ey = R * np.sin(u) * np.sin(v)
    ez = R * np.cos(v)
    ax.plot_surface(ex, ey, ez, color='blue', alpha=0.3, linewidth=0)

    # Axis labels and title
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_title("Satellite Orbit with Earth Centered")
    ax.legend()

    # Equal aspect ratio for 3D
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()

def plot2d(log_df):
    """
    Plots only the satellite's XY projection, with Earth shown as a blue circle.
    """
    x = log_df["X (km)"].values
    y = log_df["Y (km)"].values
    status = log_df["Status"].values

    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot for each status
    ax.scatter(x[status == "stationkeeping"], y[status == "stationkeeping"], label="stationkeeping", s=3, marker='^')
    ax.scatter(x[status == "non-stationkeeping"], y[status == "non-stationkeeping"], label="non-stationkeeping", s=3, marker='x')

    # Draw Earth as a circle (top-down view)
    earth = plt.Circle((0, 0), 6371, color='blue', alpha=0.2, label="Earth (not to scale)")
    ax.add_artist(earth)

    # Formatting
    ax.set_xlabel("X Position (km)")
    ax.set_ylabel("Y Position (km)")
    ax.set_title("2D XY View of Satellite Orbit")
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

# ---------- Public plotting API ----------
def plot_paths_xy(df_or_path: Union[str, pd.DataFrame],
                  earth_radius_km: float = EARTH_RADIUS_KM,
                  save_path: str | None = None,
                  show: bool = True,
                  focus: str = "geo",            # "geo" = zoom near Earth, "all" = include Sun
                  geo_margin_km: float = 50_000  # Padding when zoomed near Earth
                  ):
    """
    Plots the XY projection of Satellite, Moon, and Sun.
    Draws Earth as a blue circle and adjusts axis limits based on focus mode.
    """
    df = _as_df(df_or_path)  # Load DataFrame

    # Identify available columns
    sat_cols  = _get_cols(df, None)
    sun_cols  = _get_cols(df, "Sun")
    moon_cols = _get_cols(df, "Moon")

    fig, ax = plt.subplots(figsize=(10, 3))

    # Draw Earth circle
    earth = plt.Circle((0, 0), earth_radius_km, color='blue', alpha=0.2, label='Earth (not to scale)')
    ax.add_artist(earth)

    xs_all, ys_all = [], []  # Store for axis scaling

    # Satellite path
    if sat_cols:
        sx, sy = df[sat_cols[0]].to_numpy(), df[sat_cols[1]].to_numpy()
        ax.plot(sx, sy, linewidth=1.5, label='Satellite')
        xs_all.append(sx); ys_all.append(sy)

    # Moon path
    if moon_cols:
        mx, my = df[moon_cols[0]].to_numpy(), df[moon_cols[1]].to_numpy()
        ax.plot(mx, my, linewidth=1, label='Moon')
        xs_all.append(mx); ys_all.append(my)
        
    # Sun path
    if sun_cols:
        sunx, suny = df[sun_cols[0]].to_numpy(), df[sun_cols[1]].to_numpy()
        ax.plot(sunx, suny, linewidth=1, label='Sun')

    # Set aspect ratio and labels
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
    ax.set_title('XY Projection: Satellite, Sun, Moon (Earth as blue circle)')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

    # Adjust axis limits depending on focus mode
    if focus == "geo":
        if xs_all and ys_all:
            x_all = np.concatenate(xs_all); y_all = np.concatenate(ys_all)
            xmin, xmax = np.nanmin(x_all), np.nanmax(x_all)
            ymin, ymax = np.nanmin(y_all), np.nanmax(y_all)
            xmin = min(xmin - geo_margin_km, -earth_radius_km * 1.2)
            xmax = max(xmax + geo_margin_km,  earth_radius_km * 1.2)
            ymin = min(ymin - geo_margin_km, -earth_radius_km * 1.2)
            ymax = max(ymax + geo_margin_km,  earth_radius_km * 1.2)
            dx, dy = xmax - xmin, ymax - ymin
            m = max(dx, dy)
            cx, cy = 0.5*(xmax + xmin), 0.5*(ymax + ymin)
            ax.set_xlim(cx - m/2, cx + m/2)
            ax.set_ylim(cy - m/2, cy + m/2)
    else:
        xs_all2, ys_all2 = xs_all.copy(), ys_all.copy()
        if sun_cols:
            xs_all2.append(sunx); ys_all2.append(suny)
        if xs_all2 and ys_all2:
            x_all = np.concatenate(xs_all2); y_all = np.concatenate(ys_all2)
            xmin, xmax = np.nanmin(x_all), np.nanmax(x_all)
            ymin, ymax = np.nanmin(y_all), np.nanmax(y_all)
            dx = max(xmax - xmin, 1.0); dy = max(ymax - ymin, 1.0)
            pad = 0.05 * max(dx, dy)
            ax.set_xlim(xmin - pad, xmax + pad)
            ax.set_ylim(ymin - pad, ymax + pad)

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

def plot_paths_3d(df_or_path: Union[str, pd.DataFrame],
                  earth_radius_km: float = EARTH_RADIUS_KM,
                  save_path: str | None = None,
                  show: bool = True,
                  focus: str = "geo",
                  geo_margin_km: float = 50_000):
    """
    Plots 3D trajectories of Satellite, Moon, and Sun.
    Earth is drawn as a translucent blue sphere.
    """
    df = _as_df(df_or_path)

    # Identify columns
    sat_cols  = _get_cols(df, None)
    sun_cols  = _get_cols(df, "Sun")
    moon_cols = _get_cols(df, "Moon")

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw Earth sphere
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = earth_radius_km * np.outer(np.cos(u), np.sin(v))
    ys = earth_radius_km * np.outer(np.sin(u), np.sin(v))
    zs = earth_radius_km * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, alpha=0.15, color='blue', linewidth=0, antialiased=True)

    groups = []  # store (x, y, z, tag) for scaling

    # Satellite path
    if sat_cols:
        x, y, z = df[sat_cols[0]].to_numpy(), df[sat_cols[1]].to_numpy(), df[sat_cols[2]].to_numpy()
        ax.plot(x, y, z, linewidth=1.5, label='Satellite')
        groups.append((x, y, z, "geo"))

    # Moon path
    if moon_cols:
        x, y, z = df[moon_cols[0]].to_numpy(), df[moon_cols[1]].to_numpy(), df[moon_cols[2]].to_numpy()
        ax.plot(x, y, z, linewidth=1, label='Moon')
        groups.append((x, y, z, "geo"))

    # Sun path
    if sun_cols:
        x, y, z = df[sun_cols[0]].to_numpy(), df[sun_cols[1]].to_numpy(), df[sun_cols[2]].to_numpy()
        ax.plot(x, y, z, linewidth=1, label='Sun')
        groups.append((x, y, z, "sun"))

    # Axis limits depending on focus
    xs_sel, ys_sel, zs_sel = [], [], []
    for x, y, z, tag in groups:
        if focus == "all" or (focus == "geo" and tag == "geo"):
            xs_sel.append(x); ys_sel.append(y); zs_sel.append(z)

    if xs_sel:
        X = np.concatenate(xs_sel); Y = np.concatenate(ys_sel); Z = np.concatenate(zs_sel)
        xmin, xmax = np.nanmin(X), np.nanmax(X)
        ymin, ymax = np.nanmin(Y), np.nanmax(Y)
        zmin, zmax = np.nanmin(Z), np.nanmax(Z)
        xmin = min(xmin - geo_margin_km, -earth_radius_km * 1.2)
        xmax = max(xmax + geo_margin_km,  earth_radius_km * 1.2)
        ymin = min(ymin - geo_margin_km, -earth_radius_km * 1.2)
        ymax = max(ymax + geo_margin_km,  earth_radius_km * 1.2)
        zmin = min(zmin - geo_margin_km, -earth_radius_km * 1.2)
        zmax = max(zmax + geo_margin_km,  earth_radius_km * 1.2)
        cx, cy, cz = 0.5*(xmax+xmin), 0.5*(ymax+ymin), 0.5*(zmax+zmin)
        r = max(xmax - xmin, ymax - ymin, zmax - zmin)
        r = max(r, 1.0) * 0.5
        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy - r, cy + r)
        ax.set_zlim(cz - r, cz + r)

    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_zlabel('Z (km)')
    ax.set_title('3D Paths: Satellite, Sun, Moon (Earth as sphere)')
    ax.legend(loc='upper right')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

