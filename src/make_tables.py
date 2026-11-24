"""LaTeX tables generated directly from fig/summary.json."""
import json, sys, os
S = json.load(open(sys.argv[1]))
OUT = sys.argv[2]
NAMES = ["Raw 3-sigma", "Raw Tukey IQR", "Raw KDE quantile", "Residual 3-sigma", "Residual EWMA"]
TEX = {"Raw 3-sigma": r"Raw $3\sigma$", "Raw Tukey IQR": r"Raw Tukey fences",
       "Raw KDE quantile": r"Raw KDE quantiles", "Residual 3-sigma": r"Residual $3\sigma$",
       "Residual EWMA": r"Residual EWMA"}
CHANCE = {k: v["p_any_in_20_days"] for k, v in S["false_alarms"]["calibrated"].items()}


def f(x, d=2):
    return f"{x:.{d}f}"


# Table: false alarms
rows = []
for k in NAMES:
    a = S["false_alarms"]["default"][k]; b = S["false_alarms"]["calibrated"][k]
    rows.append(f"{TEX[k]} & {f(S['thresholds']['default'][k])} & {f(a['episodes_per_30_days'],1)} & "
                f"{f(a['p_any_in_20_days'])} & {f(S['thresholds']['calibrated'][k])} & "
                f"{f(b['episodes_per_30_days'])} & {f(b['p_any_in_20_days'])} \\\\")
open(f"{OUT}/tab_fa.tex", "w").write("\n".join(rows))

# Table: detection (calibrated)
SEV = {"wear": r"\% head coeff.", "clog": r"\% resistance", "bearing": r"vibration $\times$"}
LAB = {"wear": "Impeller wear", "clog": "Clogging", "bearing": "Bearing degradation"}


def det_rows(block, kinds):
    rows = []
    for kind in kinds:
        d = block[kind]
        rows.append(r"\multicolumn{5}{l}{\textit{" + LAB.get(kind.split("@")[0], kind) +
                    (f" (rate $\\times${kind.split('@')[1]})" if "@" in kind else "") + r"}} \\")
        for k in NAMES:
            v = d[k]
            lo, hi = v["ci"]
            above = lo > CHANCE[k]
            p = f"{f(v['p'])} [{f(lo)}, {f(hi)}]"
            if above and v["p"] < 0.5:
                dl, sv = r"$\dagger$", r"$\dagger$"
            elif above:
                dl = f"{f(v['delay_med'],1)} ({f(v['delay_q1'],1)}--{f(v['delay_q3'],1)})"
                sv = f(v["sev_med"], 2 if kind.startswith("bearing") else 1)
            else:
                dl, sv = "--", "--"
            rows.append(f"\\quad {TEX[k]} & {p} & {dl} & {sv} & {SEV[kind.split('@')[0]]} \\\\")
        rows.append(r"\addlinespace")
    return "\n".join(rows[:-1])


open(f"{OUT}/tab_det.tex", "w").write(det_rows(S["faults"], ["wear", "clog", "bearing"]))

# Table: sensitivity sweep, compact: P and median delay only
rows = []
for k in NAMES:
    cells = []
    for kind in ["wear", "clog", "bearing"]:
        for sv in ["0.5", "2.0"]:
            v = S["sweep"][f"{kind}@{sv}"][k]
            if v["ci"][0] > CHANCE[k] and v["p"] >= 0.5:
                cells.append(f"{f(v['p'])} / {f(v['delay_med'],1)}")
            elif v["ci"][0] > CHANCE[k]:
                cells.append(f"{f(v['p'])} / $\\dagger$")
            else:
                cells.append(f"{f(v['p'])} / --")
    rows.append(f"{TEX[k]} & " + " & ".join(cells) + r" \\")
open(f"{OUT}/tab_sweep.tex", "w").write("\n".join(rows))

# Regression quality
rows = []
for c, lab in [("Q", r"Flow $Q$ (system side)"), ("H", r"Head $H$"), ("P", r"Power $P$"), ("v", r"Vibration $\ln v$")]:
    rows.append(f"{lab} & {f(S['r2'][c],4)} & {f(S['resid_sd_ratio'][c],3)} \\\\")
open(f"{OUT}/tab_reg.tex", "w").write("\n".join(rows))
print("tables written")
