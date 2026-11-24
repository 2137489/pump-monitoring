"""Figures (PDF) and summary numbers for the paper."""
import json, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pumpsim as ps

OUT = sys.argv[1] if len(sys.argv) > 1 else "fig"
os.makedirs(OUT, exist_ok=True)
R = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "results.json"))

# colour palette
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
LS = [(0, (1, 1.5)), (0, (4, 1.5)), (0, (6, 2, 1, 2)), "-", "-"]
LW = [1.6, 1.6, 1.6, 1.8, 2.4]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
NAMES = list(ps.DETECTORS)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": "#c3c2b7", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.dpi": 300,
})
FAULT_LABEL = {"wear": "Impeller wear", "clog": "Clogging", "bearing": "Bearing degradation"}
SEV_UNIT = {"wear": "% head loss", "clog": "% resistance increase", "bearing": "vibration factor"}
summary = {}

# ---------------------------------------------------------------- Fig. 1
rng = np.random.default_rng(7)
m_h, tr_h = ps.simulate_run("healthy", rng)
m_w, tr_w = ps.simulate_run("wear", np.random.default_rng(8), t0=ps.TRAIN_DAYS + 5.0, sev=1.0)
n_tr = ps.N_TR
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
ax = axes[0]
ax.scatter(m_h["s"][:n_tr], m_h["H"][:n_tr], s=2, color=MUTED, alpha=0.35, lw=0, label="Healthy (training)")
late = (tr_w["t"] >= ps.TRAIN_DAYS + 25) & (tr_w["t"] < ps.TRAIN_DAYS + 30)
ax.scatter(m_w["s"][late], m_w["H"][late], s=4, color=C[1], alpha=0.8, lw=0,
           label=f"Worn impeller ({100*tr_w['wear'][late].mean():.1f}% head-coeff. loss)")
mu, sd = m_h["H"][:n_tr].mean(), m_h["H"][:n_tr].std()
for v in (mu - 3 * sd, mu + 3 * sd):
    ax.axhline(v, color=INK2, lw=0.9, ls="--")
ax.text(0.705, mu - 3 * sd + 0.6, "raw 3σ limits", color=INK2, fontsize=7.5)
ax.set_xlabel("Relative speed s = n / n$_{rated}$"); ax.set_ylabel("Head H, m")
ax.set_title("(a) Raw head: fault hidden in regime variation", loc="left")
ax.legend(loc="upper left", markerscale=3, fontsize=7.5)
ax = axes[1]
res_tr, res_te = ps.residuals(m_w)
t_te = tr_w["t"][n_tr:]
ax.plot(t_te - ps.TRAIN_DAYS, res_te["H"], color=MUTED, lw=0.4, alpha=0.6, label="Standardized residual of H")
ax.plot(t_te - ps.TRAIN_DAYS, ps.ewma(res_te["H"], 0.1) / np.sqrt(0.1 / 1.9), color=C[4], lw=1.8,
        label="EWMA of residual (scaled)")
ax.axvline(5.0, color=INK, lw=0.9, ls=":"); ax.text(5.4, 6.3, "fault onset", fontsize=7.5, color=INK)
thr = R["thresholds"]["calibrated"]["Residual EWMA"]
ax.axhline(-thr, color=INK2, lw=0.9, ls="--")
ax.text(29.5, -thr + 0.8, "calibrated alarm limit", ha="right", fontsize=7.5, color=INK2)
ax.set_xlabel("Test time, days"); ax.set_ylabel("Standardized units")
ax.set_xlim(0, 30); ax.set_ylim(-30, 8)
ax.set_title("(b) Regression residual reveals the drift", loc="left")
ax.legend(loc="lower left", fontsize=7.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_regime_masking.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig. 2
cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
raw = np.corrcoef([m_h[c][:n_tr] for c in ps.SIGNALS])
r_tr, _ = ps.residuals(m_h)
rr = np.corrcoef([r_tr[c] for c in ps.SIGNALS])
summary["corr_raw"] = np.round(raw, 2).tolist(); summary["corr_res"] = np.round(rr, 2).tolist()
lab = ["Q", "H", "P", "v"]
fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.4))
for ax, M, ttl in zip(axes, (raw, rr), ("(a) Raw signals", "(b) Regression residuals")):
    ax.imshow(M, cmap=cmap, vmin=-1, vmax=1); ax.grid(False)
    ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(lab); ax.set_yticklabels(lab)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color=INK)
    ax.set_title(ttl, loc="left")
    for sp in ax.spines.values(): sp.set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_correlations.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig. 3: cumulative detection
H = R["meta"]["horizon_days"]
grid = np.linspace(0, H, 201)
fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.4), sharey=True)
for ax, kind in zip(axes, ["wear", "clog", "bearing"]):
    res = R["faults"][kind]["calibrated"]
    for i, k in enumerate(NAMES):
        d = np.array(res[k]["delay"])
        ax.plot(grid, [(d <= g).mean() for g in grid], color=C[i], ls=LS[i], lw=LW[i], label=k)
    ax.axhline(R["meta"]["target_p_fa_20d"], color=MUTED, lw=0.8, ls="--")
    ax.set_title(FAULT_LABEL[kind], loc="left"); ax.set_xlabel("Days since fault onset")
    ax.set_xlim(0, H); ax.set_ylim(0, 1.02)
axes[0].set_ylabel("Share of runs detected")
axes[0].text(H * 0.98, 0.14, "false-alarm level", ha="right", va="bottom", fontsize=7, color=MUTED)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=5, fontsize=7.5, bbox_to_anchor=(0.5, 1.08), handlelength=3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_detection_curves.pdf"); plt.close(fig)

# ---------------------------------------------------------------- summary numbers
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def summarize(res):
    out = {}
    for k in NAMES:
        d = np.array(res[k]["delay"]); s = np.array(res[k]["sev"]); w = d <= H
        out[k] = dict(p=float(w.mean()), ci=wilson(int(w.sum()), len(d)),
                      delay_med=float(np.median(d[w])) if w.any() else None,
                      delay_q1=float(np.percentile(d[w], 25)) if w.any() else None,
                      delay_q3=float(np.percentile(d[w], 75)) if w.any() else None,
                      sev_med=float(np.nanmedian(s[w])) if w.any() else None,
                      n=len(d))
    return out


summary["faults"] = {k: summarize(R["faults"][k]["calibrated"]) for k in R["faults"]}
summary["faults_default"] = {k: summarize(R["faults"][k]["default"]) for k in R["faults"]}
summary["sweep"] = {k: summarize(v["calibrated"]) for k, v in R["sweep"].items()}
summary["false_alarms"] = R["false_alarms"]; summary["thresholds"] = R["thresholds"]
summary["meta"] = R["meta"]

# R^2 of the regime regressions on training data (mean over 20 healthy runs)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
r2 = {c: [] for c in ps.REG_MODELS}
sdr = {c: [] for c in ps.REG_MODELS}
g2 = np.random.default_rng(99)
for _ in range(20):
    m, _ = ps.simulate_run("healthy", g2)
    X = ps.regime_features({k: v[:n_tr] for k, v in m.items()})
    for c, (cols, deg) in ps.REG_MODELS.items():
        A = PolynomialFeatures(deg, include_bias=False).fit_transform(X[:, cols])
        y = np.log(m[c][:n_tr]) if c == "v" else m[c][:n_tr]
        reg = LinearRegression().fit(A, y)
        r2[c].append(reg.score(A, y))
        res = y - reg.predict(A)
        sdr[c].append(res.std() / y.std())
summary["r2"] = {c: float(np.mean(v)) for c, v in r2.items()}
summary["resid_sd_ratio"] = {c: float(np.mean(v)) for c, v in sdr.items()}
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
print(json.dumps({"r2": summary["r2"], "ratio": summary["resid_sd_ratio"]}, indent=1))
