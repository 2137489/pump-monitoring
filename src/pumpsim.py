"""Physically motivated simulator of a variable-speed centrifugal pump and
statistical detectors used in the study.

All modelling assumptions are stated explicitly in the paper (Section 2).
"""
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ----------------------------------------------------------------------------
# Pump and system parameters (rated speed s = n / n_rated = 1)
# ----------------------------------------------------------------------------
P = dict(
    H0=50.0,        # shut-off head at rated speed, m
    a=1.0e-3,       # head-curve coefficient, m/(m3/h)^2
    Q_bep=100.0,    # best-efficiency flow at rated speed, m3/h
    eta_max=0.80,   # peak hydraulic efficiency
    c_eta=0.60,     # curvature of the efficiency parabola
    eta_motor=0.92, # motor + drive efficiency
    H_st=15.0,      # mean static head of the system, m
    k=2.5e-3,       # system resistance coefficient, m/(m3/h)^2
    v0=1.2,         # healthy vibration velocity RMS at BEP and rated speed, mm/s
    v_exp=1.5,      # vibration-vs-speed exponent (assumption)
    v_off=0.5,      # vibration increase away from BEP (assumption)
    rho=1000.0, g=9.81,
)
DT_MIN = 10                         # sampling period, minutes
SPD = 24 * 60 // DT_MIN             # samples per day = 144
TRAIN_DAYS, TEST_DAYS = 30, 60
SIGNALS = ["Q", "H", "P", "v"]      # monitored sensors


def operating_regime(n, rng):
    """Speed setpoint s(t) and tank level (static-head variation) for n samples."""
    t = np.arange(n) / SPD                               # time in days
    s = np.empty(n)
    i = 0
    while i < n:                                         # piecewise-constant setpoints
        dur = rng.integers(2 * 6, 8 * 6 + 1)             # 2-8 hours (10-min samples)
        s[i:i + dur] = rng.uniform(0.72, 1.0)
        i += dur
    s = np.clip(s + 0.02 * np.sin(2 * np.pi * t), 0.70, 1.0)   # small daily pattern
    level = 1.0 * np.sin(2 * np.pi * (t - 0.25)) + np.cumsum(rng.normal(0, 0.005, n))
    level -= np.mean(level)
    return t, s, level


def pump_physics(s, level, wear, clog, bearing):
    """True (noise-free) operating point for given regime and fault state."""
    H_st = P["H_st"] + level
    k = P["k"] * (1.0 + clog)
    h0 = (1.0 - wear) * P["H0"] * s ** 2
    aw = (1.0 - wear) * P["a"]
    Q = np.sqrt(np.maximum(h0 - H_st, 0.0) / (aw + k))           # pump curve = system curve
    H = H_st + k * Q ** 2
    x = (Q / s - P["Q_bep"]) / P["Q_bep"]                         # distance from BEP (affinity-scaled)
    eta = P["eta_max"] * (1.0 - 0.5 * wear) * (1.0 - P["c_eta"] * x ** 2)
    eta = np.clip(eta, 0.30, None)
    P_el = P["rho"] * P["g"] * (Q / 3600.0) * H / eta / P["eta_motor"] / 1000.0   # kW
    v = P["v0"] * s ** P["v_exp"] * (1.0 + P["v_off"] * x ** 2) * bearing
    return Q, H, P_el, v


def fault_profile(kind, t, t0, sev=1.0):
    """Fault state trajectories; zero before onset t0 (days). sev scales magnitude."""
    tau = np.clip(t - t0, 0, None)
    wear = np.zeros_like(t); clog = np.zeros_like(t); bearing = np.ones_like(t)
    if kind == "wear":        # impeller wear: head coefficient drops 6 % in 25 days
        wear = sev * 0.06 * tau / 25.0
    elif kind == "clog":      # strainer/valve clogging: resistance +40 % in 25 days
        clog = sev * 0.40 * tau / 25.0
    elif kind == "bearing":   # exponential bearing degradation: x2.5 vibration in 25 days
        beta = 0.15
        A = 1.5 / (np.exp(beta * 25.0) - 1.0)
        bearing = 1.0 + sev * A * (np.exp(beta * tau) - 1.0)
    return wear, clog, bearing


def simulate_run(kind, rng, t0=None, sev=1.0):
    """One 90-day run: 30 healthy training days + 60 test days."""
    n = (TRAIN_DAYS + TEST_DAYS) * SPD
    t, s, level = operating_regime(n, rng)
    if kind == "healthy":
        t0 = np.inf
    elif t0 is None:
        t0 = TRAIN_DAYS + rng.uniform(5, 35)
    wear, clog, bearing = fault_profile(kind, t, t0, sev)
    Q, H, P_el, v = pump_physics(s, level, wear, clog, bearing)
    meas = dict(
        s=s * (1 + rng.normal(0, 0.002, n)),
        level=level + rng.normal(0, 0.05, n),
        Q=Q * (1 + rng.normal(0, 0.005, n)),
        H=H * (1 + rng.normal(0, 0.005, n)) + rng.normal(0, 0.1, n),
        P=P_el * (1 + rng.normal(0, 0.01, n)),
        v=v * np.exp(rng.normal(0, 0.08, n)),
    )
    truth = dict(t=t, t0=t0, wear=wear, clog=clog, bearing=bearing,
                 Q=Q, H=H, P=P_el, v=v)
    return meas, truth


# ----------------------------------------------------------------------------
# Detectors. Every detector returns a per-sample score on the test part and a
# default threshold taken from the textbook rule; alarm = score > threshold.
# ----------------------------------------------------------------------------
N_TR = TRAIN_DAYS * SPD


def _split(meas):
    tr = {k: v[:N_TR] for k, v in meas.items()}
    te = {k: v[N_TR:] for k, v in meas.items()}
    return tr, te


def score_raw_3sigma(meas):
    tr, te = _split(meas)
    z = [np.abs(te[c] - tr[c].mean()) / tr[c].std() for c in SIGNALS]
    return np.max(z, axis=0), 3.0


def score_raw_iqr(meas):
    tr, te = _split(meas)
    d = []
    for c in SIGNALS:
        q1, q3 = np.percentile(tr[c], [25, 75]); iqr = q3 - q1
        d.append(np.maximum((q1 - te[c]) / iqr, (te[c] - q3) / iqr))
    return np.max(d, axis=0), 1.5          # Tukey fences


def score_raw_kde(meas, grid=512):
    """Score = -log10 of the two-sided tail probability under a KDE of training data."""
    tr, te = _split(meas)
    sc = []
    for c in SIGNALS:
        x = tr[c]
        kde = gaussian_kde(x)
        lo, hi = x.min() - 4 * x.std(), x.max() + 4 * x.std()
        g = np.linspace(lo, hi, grid)
        cdf = np.cumsum(kde(g)); cdf /= cdf[-1]
        F = np.interp(te[c], g, cdf, left=0.0, right=1.0)
        tail = np.clip(np.minimum(F, 1 - F), 1e-12, None)
        sc.append(-np.log10(tail))
    return np.max(sc, axis=0), -np.log10(0.0005)   # 0.05 % / 99.95 % percentage points


def regime_features(m):
    """Operating-regime features used as regressors (speed, flow, tank level)."""
    return np.column_stack([m["s"], m["Q"], m["level"]])


REG_MODELS = {   # target: (feature columns, polynomial degree)
    "H": ([0, 1, 2], 2),       # pump-side head:    H = f(s, Q, level)
    "P": ([0, 1, 2], 3),       # pump-side power:   P = f(s, Q, level)
    "v": ([0, 1, 2], 2),       # vibration:         v = f(s, Q, level)
    "Q": ([0, 2], 2),          # system side flow:  Q = f(s, level)
}


def residuals(meas):
    tr, te = _split(meas)
    Xtr, Xte = regime_features(tr), regime_features(te)
    out_tr, out_te = {}, {}
    for c, (cols, deg) in REG_MODELS.items():
        pf = PolynomialFeatures(deg, include_bias=False)
        A = pf.fit_transform(Xtr[:, cols]); B = pf.transform(Xte[:, cols])
        ytr = np.log(tr[c]) if c == "v" else tr[c]
        yte = np.log(te[c]) if c == "v" else te[c]
        reg = LinearRegression().fit(A, ytr)
        rtr = ytr - reg.predict(A); rte = yte - reg.predict(B)
        sd = rtr.std()
        out_tr[c], out_te[c] = rtr / sd, rte / sd
    return out_tr, out_te


def score_res_3sigma(meas, res=None):
    _, rte = res if res is not None else residuals(meas)
    return np.max([np.abs(rte[c]) for c in SIGNALS], axis=0), 3.0


def ewma(x, lam):
    """Exponentially weighted moving average z_t = lam*x_t + (1-lam)*z_{t-1}, z_0 = 0."""
    from scipy.signal import lfilter
    return lfilter([lam], [1.0, -(1.0 - lam)], x)


def score_res_ewma(meas, res=None, lam=0.1):
    _, rte = res if res is not None else residuals(meas)
    k = np.sqrt(lam / (2 - lam))            # asymptotic std of EWMA of N(0,1)
    return np.max([np.abs(ewma(rte[c], lam)) / k for c in SIGNALS], axis=0), 3.0


DETECTORS = {
    "Raw 3-sigma": score_raw_3sigma,
    "Raw Tukey IQR": score_raw_iqr,
    "Raw KDE quantile": score_raw_kde,
    "Residual 3-sigma": score_res_3sigma,
    "Residual EWMA": score_res_ewma,
}


def all_scores(meas):
    res = residuals(meas)
    out = {}
    for name, f in DETECTORS.items():
        out[name] = f(meas, res) if name.startswith("Residual") else f(meas)
    return out
