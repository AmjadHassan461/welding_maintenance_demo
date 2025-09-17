from pathlib import Path
import numpy as np
import pandas as pd
from src.config import (
    SENSORS_RAW_DIR,
    SENSOR_WINDOW,
    SENSOR_STEP,
    SENSOR_FEATURES,
    RANDOM_STATE,
)

def _window_array(arr, window, step):
    starts = np.arange(0, len(arr) - window + 1, step)
    return np.stack([arr[s:s+window] for s in starts], axis=0) if len(starts) > 0 else np.empty((0, window) + arr.shape[1:])

def _synth_sensor_df(n_samples=1000, n_faults=3, seed=RANDOM_STATE):
    """
    Synthetic fallback (kept for consistency). Not used if real CSV exists.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples)

    vib = 0.5*np.sin(2*np.pi*0.01*t) + 0.1*rng.standard_normal(n_samples)
    cur = 5 + 0.5*np.sin(2*np.pi*0.03*t) + 0.2*rng.standard_normal(n_samples)
    tmp = 40 + 0.05*t + 0.5*rng.standard_normal(n_samples)
    label = np.zeros(n_samples, dtype=int)

    fault_positions = np.linspace(200, n_samples-200, n_faults, dtype=int)
    for start in fault_positions:
        end = min(start+200, n_samples)
        vib[start:end] += 0.5
        cur[start:end] += 0.5
        tmp[start:end] += 5
        label[start:end] = 1

    return pd.DataFrame({"vibration": vib, "current": cur, "temp": tmp, "label": label})

def _load_real_tabular():
    # Look for CSV files in the specified directory
    csvs = list(Path(SENSORS_RAW_DIR).glob("*.csv"))
    if not csvs:
        return None, None

    # Use the first matching CSV (or concatenate matching schemas)
    dfs = []
    for p in csvs:
        df = pd.read_csv(p)
        if all(c in df.columns for c in SENSOR_FEATURES) and "Defect" in df.columns:
            dfs.append(df[SENSOR_FEATURES + ["Defect"]])
    if not dfs:
        return None, None

    df = pd.concat(dfs, ignore_index=True).dropna(subset=SENSOR_FEATURES + ["Defect"])
    X = df[SENSOR_FEATURES].astype("float32").values
    y = df["Defect"].astype(int).values
    return X, y

def load_sensor_windows():
    X_tab, y_tab = _load_real_tabular()
    if X_tab is not None:
        # Wrap each sample as a 1-length window
        Xw = X_tab[:, None, :]  # [N, 1, F]
        yw = y_tab
        return Xw, yw

    # Fallback: synthetic sequence data into windows (legacy behavior)
    df = _synth_sensor_df()
    X = df[["vibration", "current", "temp"]].values
    y = df["label"].values
    Xw = _window_array(X, SENSOR_WINDOW, SENSOR_STEP)           # [N, W, F]
    yw = _window_array(y, SENSOR_WINDOW, SENSOR_STEP)[:, -1]    # label at window end
    return Xw, yw
