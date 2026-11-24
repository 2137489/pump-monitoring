"""Monte Carlo experiments: false-alarm rates and fault detection performance."""
import json, sys, time
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import pumpsim as ps

SEED = 2026
N_CAL, N_HEALTHY, N_FAULT, N_SWEEP = 200, 200, 200, 100
FAULTS = ["wear", "clog", "bearing"]
SWEEP = [0.5, 2.0]                   # fault-rate multipliers for the sensitivity study
TARGET_P_FA = 0.10                   # P(at least one false alarm in a 20-day window)
HORIZON = 20.0                       # days after onset used for detection probability
names = list(ps.DETECTORS)
rng = np.random.default_rng(SEED)
t_test = np.arange(ps.TEST_DAYS * ps.SPD) / ps.SPD + ps.TRAIN_DAYS   # days


def scores_of(kind, sev=1.0):
    m, tr = ps.simulate_run(kind, rng, sev=sev)
    sc = ps.all_scores(m)
    return {k: v[0].astype(np.float32) for k, v in sc.items()}, tr


t_start = time.time()
# 1) calibration runs (healthy): threshold = 90 % quantile of the maximum score
#    over non-overlapping 20-day windows -> P(false alarm in 20 days) = 10 %
W = int(HORIZON * ps.SPD)
n_win = (ps.TEST_DAYS * ps.SPD) // W


def window_max(x):
    return x[:n_win * W].reshape(n_win, W).max(axis=1)


cal = {k: [] for k in names}
for _ in range(N_CAL):
    s, _ = scores_of("healthy")
    for k in names:
        cal[k].append(window_max(s[k]))
default_thr = {"Raw 3-sigma": 3.0, "Raw Tukey IQR": 1.5, "Raw KDE quantile": float(-np.log10(0.0005)),
               "Residual 3-sigma": 3.0, "Residual EWMA": 3.0}
calib_thr = {k: float(np.quantile(np.concatenate(cal[k]), 1 - TARGET_P_FA)) for k in names}

# 2) independent healthy runs -> false alarm statistics
fa = {mode: {k: {"samples": [], "win": []} for k in names} for mode in ("default", "calibrated")}
for _ in range(N_HEALTHY):
    s, _ = scores_of("healthy")
    for k in names:
        wm = window_max(s[k])
        for mode, thr in (("default", default_thr), ("calibrated", calib_thr)):
            al = s[k] > thr[k]
            events = int(al[0]) + int((al[1:] & ~al[:-1]).sum())   # alarm episodes
            fa[mode][k]["samples"].append(events)
            fa[mode][k]["win"].extend((wm > thr[k]).tolist())


def severity_at(kind, tr, idx):
    """Physical fault magnitude at sample idx (test indexing)."""
    j = ps.N_TR + idx
    if kind == "wear":
        return 100 * tr["wear"][j]                    # % head-coefficient loss
    if kind == "clog":
        return 100 * tr["clog"][j]                    # % increase of system resistance
    return tr["bearing"][j]                           # vibration multiplier


def detect(kind, sev, n_runs):
    res = {mode: {k: {"delay": [], "sev": [], "early": 0} for k in names}
           for mode in ("default", "calibrated")}
    for _ in range(n_runs):
        s, tr = scores_of(kind, sev)
        t0 = tr["t0"]
        after = t_test >= t0
        for mode, thr in (("default", default_thr), ("calibrated", calib_thr)):
            for k in names:
                al = s[k] > thr[k]
                if al[~after].any():
                    res[mode][k]["early"] += 1          # alarm before onset (false alarm)
                hit = np.flatnonzero(al & after)
                if hit.size:
                    d = t_test[hit[0]] - t0
                    res[mode][k]["delay"].append(float(d))
                    res[mode][k]["sev"].append(float(severity_at(kind, tr, hit[0])))
                else:
                    res[mode][k]["delay"].append(float("inf"))
                    res[mode][k]["sev"].append(float("nan"))
    return res


results = {"thresholds": {"default": default_thr, "calibrated": calib_thr},
           "false_alarms": {}, "faults": {}, "sweep": {}}
for mode in fa:
    results["false_alarms"][mode] = {
        k: {"episodes_per_30_days": float(np.mean(fa[mode][k]["samples"]) / ps.TEST_DAYS * 30),
            "p_any_in_20_days": float(np.mean(fa[mode][k]["win"]))}
        for k in names}

for kind in FAULTS:
    results["faults"][kind] = detect(kind, 1.0, N_FAULT)
    for sv in SWEEP:
        results["sweep"][f"{kind}@{sv}"] = detect(kind, sv, N_SWEEP)

results["meta"] = dict(seed=SEED, n_cal=N_CAL, n_healthy=N_HEALTHY, n_fault=N_FAULT,
                       n_sweep=N_SWEEP, target_p_fa_20d=TARGET_P_FA,
                       horizon_days=HORIZON, runtime_s=round(time.time() - t_start, 1))
json.dump(results, open(sys.argv[1] if len(sys.argv) > 1 else "results.json", "w"), indent=1)
print("done in", results["meta"]["runtime_s"], "s")
