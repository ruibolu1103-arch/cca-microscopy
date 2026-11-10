"""Build every table, figure and quoted number of the manuscript from results/*.json (CPU only).

Run from the project root:  OMP_NUM_THREADS=2 python paper/make_assets.py
Outputs: paper/tables/*.tex, paper/figures/*.pdf|png, paper/numbers.json.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper")
TAB = os.path.join(OUT, "tables")
FIG = os.path.join(OUT, "figures")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
                     "mathtext.fontset": "stix", "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
                     "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "0.25",
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "xtick.color": "0.2", "ytick.color": "0.2", "axes.grid": True, "axes.grid.axis": "y", "grid.color": "0.9",
                     "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 7.5})
# Okabe-Ito colour-blind safe palette
C = dict(cca="#1F5FA8", comir="#E69F00", mi="#009E73", full_xa="#CC79A7", full_mis="#D55E00", unet_mis="#56B4E9",
         none="#7f7f7f", oracle="#000000")


def load(name):
    return json.load(open(os.path.join(RES, name)))


final = load("final_tissuenet.json")
comir3 = load("comir3seeds_tissuenet.json")
comir0 = load("comir_tissuenet.json")
cdead = load("comir_dead_tissuenet.json")
abl = load("ablation_tissuenet.json")
cps = load("cpsam_cca_tissuenet_test500.json")
frame = load("frame_test_tissuenet_test.json")
frame_val = load("frame_test_tissuenet.json")
rng = load("range_tissuenet.json")
fail = load("fail_tissuenet.json")
real = load("real_exemplar001.json")
dl = load("final_deepliif.json")
bench = load("bench.json")
qual = load("qual_tissuenet.json")

PS = final["per_scene"]
SCENES = list(PS.keys())
size = lambda k: int(k.split("/s")[1])
stream = lambda k: int(k.split("@")[1].split("/")[0])
MIS = [k for k in SCENES if size(k) > 0]
ALI = [k for k in SCENES if size(k) == 0]
NUC = [k for k in MIS if stream(k) == 0]
MEM = [k for k in MIS if stream(k) == 1]
assert len(MIS) == 12 and len(ALI) == 8 and len(NUC) == 6


def f1(method, k):
    if method == "comir":
        return float(np.mean([comir3["per_scene"][k][f"comir{s}"] for s in range(3)]))
    if method == "cca":
        return PS[k]["cca_mean"]
    if method == "oracle" and size(k) == 0:
        return PS[k]["none"]
    return PS[k][method]


def mean(method, scenes):
    return float(np.mean([f1(method, k) for k in scenes]))


METHODS = ["none", "full_mis", "unet_mis", "full_xa", "mi", "comir", "cca", "oracle"]
NAMES = dict(none="No alignment", full_mis="full\\_mis", unet_mis="unet\\_mis", full_xa="full\\_xa", mi="Gated MI",
             comir="CoMIR (3 seeds)", cca="CCA (3 seeds)", oracle="Oracle shift")
N = {}  # every quoted number


def gain(m, scenes):
    return (mean(m, scenes) - mean("none", scenes)) / (mean("oracle", scenes) - mean("none", scenes))


for m in METHODS:
    N[f"{m}/mis"] = mean(m, MIS)
    N[f"{m}/nuc"] = mean(m, NUC)
    N[f"{m}/mem"] = mean(m, MEM)
    N[f"{m}/ali"] = mean(m, ALI)
    N[f"{m}/harm"] = mean("none", ALI) - mean(m, ALI)
    N[f"{m}/G"] = 100 * gain(m, MIS)
    N[f"{m}/G_nuc"] = 100 * gain(m, NUC)
    N[f"{m}/G_mem"] = 100 * gain(m, MEM)
N["gap"] = N["oracle/mis"] - N["none/mis"]
N["gap_nuc"] = N["oracle/nuc"] - N["none/nuc"]
N["gap_mem"] = N["oracle/mem"] - N["none/mem"]
for k in final["point"]:
    N[f"final/{k}"] = final["point"][k]
    N[f"final/{k}/ci"] = final["ci"][k]
for s, v in final["per_seed"].items():
    for k, x in v.items():
        N[f"seed/{s}/{k}"] = x
N["comir3/mis_diff"], N["comir3/mis_diff/ci"] = comir3["point"]["cca_minus_comir"], comir3["ci"]["cca_minus_comir"]
N["comir3/ali_diff"], N["comir3/ali_diff/ci"] = comir3["point"]["cca_minus_comir_aligned"], comir3["ci"]["cca_minus_comir_aligned"]
N["comir0/mis_diff"], N["comir0/mis_diff/ci"] = comir0["point"]["cca_minus_comir"], comir0["ci"]["cca_minus_comir"]
N["comir0/ali_diff"], N["comir0/ali_diff/ci"] = comir0["point"]["cca_minus_comir_aligned"], comir0["ci"]["cca_minus_comir_aligned"]
# CoMIR seed 0 with the same +-1 px dead zone as CCA (post hoc check; CCA seed 0 as in comir0)
cca0_ali = float(np.mean([PS[k]["cca0"] for k in ALI]))
N["comir_dead/ali"] = float(np.mean([cdead["per_scene"][k]["comir_dead"] for k in ALI]))
N["comir_dead/ali_diff"] = cca0_ali - N["comir_dead/ali"]
N["comir0/ali"] = float(np.mean([comir3["per_scene"][k]["comir0"] for k in ALI]))
N["comir0/mis"] = float(np.mean([comir3["per_scene"][k]["comir0"] for k in MIS]))
N["comir_dead/mis"] = float(np.mean([cdead["per_scene"][k]["comir_dead"] for k in MIS]))
N["cca0/mis"] = float(np.mean([PS[k]["cca0"] for k in MIS]))
N["cca0/ali"] = cca0_ali


# fraction of shift errors <= 2 px on aligned scenes (per seed range)
def within2(src, prefix, scenes):
    return float(np.mean([src["shift_error"][f"{prefix}/{k}"]["within2"] for k in scenes]))


ALI_SE = [k for k in ALI if f"cca0/{k}" in final["shift_error"]]
N["within2_ali/cca"] = [within2(final, f"cca{s}", ALI_SE) for s in range(3)]
N["within2_ali/comir"] = [within2(comir3, f"comir{s}", ALI_SE) for s in range(3)]
N["within2_mis/cca"] = [within2(final, f"cca{s}", MIS) for s in range(3)]
N["within2_mis/comir"] = [within2(comir3, f"comir{s}", MIS) for s in range(3)]
N["comir_seed_mis"] = [float(np.mean([comir3["per_scene"][k][f"comir{s}"] for k in MIS])) for s in range(3)]
N["cca_seed_mis"] = [float(np.mean([PS[k][f"cca{s}"] for k in MIS])) for s in range(3)]
N["comir_seed_harm"] = [N["none/ali"] - float(np.mean([comir3["per_scene"][k][f"comir{s}"] for k in ALI])) for s in range(3)]

# Cellpose-SAM front end (500 test images, CCA seed 0)
CPS = cps["per_scene"]
cmis = [k for k in CPS if size(k) > 0]
cali = [k for k in CPS if size(k) == 0]
cm = lambda m, sc: float(np.mean([100 * CPS[k][m]["inst_f1"] for k in sc]))
for m in ("none", "cca", "comir"):
    N[f"cps/{m}/mis"], N[f"cps/{m}/ali"] = cm(m, cmis), cm(m, cali)
    N[f"cps/{m}/harm"] = cm("none", cali) - cm(m, cali)
N["cps/oracle/mis"] = cm("oracle", cmis)
for m in ("cca", "comir"):
    N[f"cps/{m}/G"] = 100 * (N[f"cps/{m}/mis"] - N["cps/none/mis"]) / (N["cps/oracle/mis"] - N["cps/none/mis"])
N["cps/harm_ci"] = cps["harm_ci"]
N["cps/gain"], N["cps/gain_ci"] = cps["gain"], cps["gain_ci"]
N["cps/vs_comir"], N["cps/vs_comir/ci"] = cps["comir"]["cca_minus_comir"], cps["comir"]["ci"]
N["cps/vs_comir_ali"], N["cps/vs_comir_ali/ci"] = cps["comir"]["cca_minus_comir_aligned"], cps["comir"]["aligned_ci"]

# Ablation (seed 0)
VARS = ["coarse", "single", "tta", "tta_dead", "short"]
AP = abl["per_scene"]
for v in VARS:
    N[f"abl/{v}/mis"] = float(np.mean([AP[k][v] for k in MIS]))
    N[f"abl/{v}/ali"] = float(np.mean([AP[k][v] for k in ALI]))
    N[f"abl/{v}/harm"] = float(np.mean([AP[k]["none"] for k in ALI])) - N[f"abl/{v}/ali"]
    N[f"abl/{v}/w2"] = float(np.mean([abl["shift_error"][f"{v}/{k}"]["within2"] for k in MIS]))
    N[f"abl/{v}/w2_ali"] = float(np.mean([abl["shift_error"][f"{v}/{k}"]["within2"] for k in ALI if f"{v}/{k}" in abl["shift_error"]]))
    if v != "tta_dead":
        N[f"abl/{v}/d"] = abl["point"][f"{v}/mis_vs_adopted"]
        N[f"abl/{v}/d_ci"] = abl["ci"][f"{v}/mis_vs_adopted"]
        N[f"abl/{v}/da"] = abl["point"][f"{v}/ali_vs_adopted"]
        N[f"abl/{v}/da_ci"] = abl["ci"][f"{v}/ali_vs_adopted"]

# Frame test (test set, full seed 0)
N["frame/clean"] = {k: v for k, v in frame["clean"].items()}
for t in (0, 1):
    for s in (4, 8, 16):
        for mode in ("fixed", "moved"):
            N[f"frame/{t}/s{s}/{mode}"] = frame[f"{t}/s{s}"][mode]
N["frame/residual"] = [100 * (frame["clean"]["inst_f1"] - frame[f"0/s{s}"]["moved"]["inst_f1"]) for s in (4, 8, 16)]

# Operating range (seed 0 CCA, both moved streams averaged)
SIZES = rng["sizes"]
for m in ("cca", "comir", "mi"):
    N[f"range/{m}"] = [float(np.mean([rng["results"][f"{m}/t{t}/s{s}"]["within2"] for t in (0, 1)])) for s in SIZES]
    N[f"range/{m}/t0"] = [rng["results"][f"{m}/t0/s{s}"]["within2"] for s in SIZES]
    N[f"range/{m}/t1"] = [rng["results"][f"{m}/t1/s{s}"]["within2"] for s in SIZES]

# Failure analysis (clean, 16 px)
for t in ("t0", "t1"):
    F = fail[t]
    N[f"fail/{t}"] = dict(rate=F["fail_rate"], all3=F["all_three_fail"], overlap=F["cca_fail_also_comir"],
                         cells_fail=F["median_when_cca_fails"]["cells"], cells_ok=F["median_when_cca_ok"]["cells"],
                         q_cca=F["cca_fail_by_quartile"]["cells"], q_comir=F["comir_fail_by_quartile"]["cells"],
                         contrast_fail=F["median_when_cca_fails"]["moving_contrast"], contrast_ok=F["median_when_cca_ok"]["moving_contrast"])

# Real cross-cycle data
G = real["groups"]


def rw(group, m, key="within2"):
    if m == "cca":
        return float(np.mean([G[group][f"cca{s}"][key] for s in range(3)]))
    return G[group][m][key]


for grp in ("cross_moved", "cross_s0", "within"):
    for m in ("none", "ncc", "mi", "cca"):
        N[f"real/{grp}/{m}"] = rw(grp, m)
    N[f"real/{grp}/n"] = G[grp]["mi"]["n"]
N["real/cca_seeds"] = [G["cross_moved"][f"cca{s}"]["within2"] for s in range(3)]
N["real/R1"], N["real/R1_ci"], N["real/R2"] = real["R1"]["within2_gain_vs_mi"], real["R1"]["ci"], real["R2"]["cca_not_shifted_within"]
N["real/dna"] = real["dna_windows"]
MARKERS = sorted({k.split("/")[1] for k in G if k.startswith("cross_moved/")})
for mk in MARKERS:
    N[f"real/m/{mk}"] = dict(n=G[f"cross_moved/{mk}"]["mi"]["n"], mi=rw(f"cross_moved/{mk}", "mi"), cca=rw(f"cross_moved/{mk}", "cca"),
                            n_within=G[f"within/{mk}"]["mi"]["n"], mi_within=rw(f"within/{mk}", "mi"), cca_within=rw(f"within/{mk}", "cca"))

# DeepLIIF
for k in dl["point"]:
    N[f"dl/{k}"], N[f"dl/{k}/ci"] = dl["point"][k], dl["ci"][k]
DPS = dl["per_scene"]
dmean = lambda m, sc: float(np.mean([DPS[k][m] for k in sc if m in DPS[k]]))
for m in ("none", "mi", "full_xa", "cca_mean", "oracle"):
    N[f"dl/{m}/mis"] = dmean(m, MIS)
    if m != "oracle":
        N[f"dl/{m}/ali"] = dmean(m, ALI)

N["bench"] = bench
N["qual"] = dict(scene_means=qual["scene_means"], mi_applied=qual["mi_applied_aligned"],
                 picks={p["tag"]: {k: v["inst_f1"] for k, v in p["methods"].items()} for p in qual["picks"]})
json.dump(N, open(os.path.join(OUT, "numbers.json"), "w"), indent=1)


# ------------------------------------------------------------------ tables
def ci(v, d=2):
    return f"[{v[0]:.{d}f}, {v[1]:.{d}f}]".replace("-", "$-$")


def sgn(x, d=2):
    return f"{x:+.{d}f}".replace("-", "$-$")


def write(name, s):
    open(os.path.join(TAB, name), "w", encoding="utf-8").write(s)


def bold_best(rows, cols, eligible, better):
    """Bold the best displayed value per column among eligible rows (ties all bold). rows: {key: [cells]}."""
    num = lambda c: float(c.replace("$-$", "-").split(" ")[0])
    for c in cols:
        vals = {r: num(rows[r][c]) for r in eligible if rows[r][c] != "--"}
        if not vals:
            continue
        best = max(vals.values()) if better[c] == "max" else min(vals.values())
        for r, v in vals.items():
            if v == best:
                rows[r][c] = f"\\textbf{{{rows[r][c]}}}"


# Table 2: main comparison
diff = dict(none=("vs_none",), full_mis=("vs_full_mis",), unet_mis=("vs_unet_mis",), full_xa=("vs_full_xa",), mi=("vs_mi",))
T = {}
for m in METHODS:
    if m == "comir":
        d = f"{sgn(N['comir3/mis_diff'])} {ci(N['comir3/mis_diff/ci'])}"
    elif m in diff:
        k = diff[m][0]
        d = f"{sgn(final['point'][k])} {ci(final['ci'][k])}"
    elif m == "oracle":
        d = f"{sgn(-final['point']['room'])} {ci([-final['ci']['room'][1], -final['ci']['room'][0]])}"
    else:
        d = "--"
    harm = "--" if m in ("none", "oracle") else f"{N[m + '/harm']:.2f}"
    G_ = "0" if m == "none" else ("100" if m == "oracle" else f"{N[m + '/G']:.1f}")
    ali = "--" if m == "oracle" else f"{N[m + '/ali']:.2f}"
    T[m] = [NAMES[m], f"{N[m + '/mis']:.2f}", f"{N[m + '/nuc']:.2f}", f"{N[m + '/mem']:.2f}", G_, ali, harm, d]
ALIGNERS = ["full_mis", "unet_mis", "full_xa", "mi", "comir", "cca"]
bold_best(T, [1, 2, 3, 4, 5, 6], ALIGNERS, {1: "max", 2: "max", 3: "max", 4: "max", 5: "max", 6: "min"})
rows = []
for m in METHODS:
    rows.append(" & ".join(T[m]) + " \\\\")
    if m in ("none", "cca"):
        rows.append("\\midrule")
write("table2_main.tex", "\n".join([
    "\\begin{tabular}{@{}lccccccc@{}}", "\\toprule",
    " & \\multicolumn{4}{c}{Misaligned scenes} & \\multicolumn{2}{c}{Aligned scenes} & \\\\",
    "\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}",
    "Method & All (12) & Nuclear moved & Membrane moved & $G$ (\\%) & F1 (8) & Harm $H$ & CCA $-$ method [95\\% CI] \\\\", "\\midrule",
    *rows, "\\bottomrule", "\\end{tabular}"]))

# Table 3: (a) CoMIR, (b) Cellpose-SAM
def block(prefix, label, comir_name, cca_name):
    g = (lambda k: N[k]) if prefix == "" else (lambda k: N["cps/" + k])
    R = {"none": ["No alignment", f"{g('none/mis'):.2f}", "0", f"{g('none/ali'):.2f}", "--"],
         "comir": [comir_name, f"{g('comir/mis'):.2f}", f"{g('comir/G'):.1f}", f"{g('comir/ali'):.2f}", f"{g('comir/harm'):.2f}"],
         "cca": [cca_name, f"{g('cca/mis'):.2f}", f"{g('cca/G'):.1f}", f"{g('cca/ali'):.2f}", f"{g('cca/harm'):.2f}"],
         "oracle": ["Oracle shift", f"{g('oracle/mis'):.2f}", "100", "--", "--"]}
    bold_best(R, [1, 2, 3, 4], ["comir", "cca"], {1: "max", 2: "max", 3: "max", 4: "min"})
    out = []
    for i, k in enumerate(("none", "comir", "cca", "oracle")):
        head = f"\\multirow{{4}}{{*}}{{{label}}}" if i == 0 else ""
        out.append(head + " & " + " & ".join(R[k]) + " \\\\")
    return out


t3 = ["\\begin{tabular}{@{}llcccc@{}}", "\\toprule",
      "Segmenter & Front end & Misaligned F1 & $G$ (\\%) & Aligned F1 & Harm $H$ \\\\", "\\midrule",
      *block("", "Dual-stream (1324 images)", "CoMIR (3 seeds)", "CCA (3 seeds)"), "\\midrule",
      *block("cps", "Cellpose-SAM (500 images)", "CoMIR (seed 0)", "CCA (seed 0)"),
      "\\bottomrule", "\\end{tabular}"]
write("table3_comir_cpsam.tex", "\n".join(t3))

# Table 4: ablation
AN = dict(coarse="Coarse level only", single="Coarse + fine, single view", tta="+ four-rotation median",
          tta_dead="+ $\\pm$1 px dead zone (adopted)", short="Adopted, 6000 iterations")
T = {}
for v in VARS:
    d = "--" if v == "tta_dead" else f"{sgn(N[f'abl/{v}/d'])} {ci(N[f'abl/{v}/d_ci'])}"
    T[v] = [AN[v], f"{N[f'abl/{v}/mis']:.2f}", f"{100 * N[f'abl/{v}/w2']:.1f}", f"{N[f'abl/{v}/harm']:.2f}", d]
bold_best(T, [1, 2, 3], VARS, {1: "max", 2: "max", 3: "min"})
write("table4_ablation.tex", "\n".join(["\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
                                         "Variant & Misaligned F1 & Error $\\le$2 px (\\%) & Harm $H$ & $\\Delta$ misaligned vs adopted [95\\% CI] \\\\",
                                         "\\midrule", *[" & ".join(T[v]) + " \\\\" for v in VARS], "\\bottomrule", "\\end{tabular}"]))

# Table 5: real cross-cycle data
RN = dict(none="No alignment", ncc="Masked NCC", mi="Gated MI", cca="CCA (3 seeds)")
T = {m: [RN[m], f"{100 * N[f'real/cross_moved/{m}']:.1f}", f"{100 * N[f'real/cross_s0/{m}']:.1f}", f"{100 * N[f'real/within/{m}']:.1f}"]
     for m in ("none", "ncc", "mi", "cca")}
bold_best(T, [1, 2, 3], ["ncc", "mi", "cca"], {1: "max", 2: "max", 3: "max"})
write("table5_real.tex", "\n".join(["\\begin{tabular}{@{}lccc@{}}", "\\toprule",
                                     f" & Cross-cycle, residual $>$0 & Cross-cycle, residual 0 & Same-cycle \\\\",
                                     f"Method & ({N['real/cross_moved/n']} pairs) & ({N['real/cross_s0/n']} pairs) & ({N['real/within/n']} pairs) \\\\",
                                     "\\midrule", *[" & ".join(T[m]) + " \\\\" for m in ("none", "ncc", "mi", "cca")], "\\midrule",
                                     "Pre-specified criterion & CCA $-$ MI $\\ge$ +10 & -- & CCA $\\ge$ 90 \\\\",
                                     "\\bottomrule", "\\end{tabular}"]))

# Table 6: efficiency
b = bench
write("table6_efficiency.tex", "\n".join([
    "\\begin{tabular}{@{}lccc@{}}", "\\toprule",
    "Component & Trainable / total parameters & ms per pair (batch 1) & ms per pair (batch 16) \\\\", "\\midrule",
    f"CCA, single view & {b['params']['cca_trainable'] / 1e6:.2f} M / {b['params']['cca_total'] / 1e6:.1f} M & {b['ms_per_pair']['cca_single/b1']:.1f} & {b['ms_per_pair']['cca_single/b16']:.1f} \\\\",
    f"CCA, adopted (four rotations) & {b['params']['cca_trainable'] / 1e6:.2f} M / {b['params']['cca_total'] / 1e6:.1f} M & {b['ms_per_pair']['cca_tta/b1']:.1f} & {b['ms_per_pair']['cca_tta/b16']:.1f} \\\\",
    f"Dual-stream segmenter & -- / {b['params']['full_total'] / 1e6:.1f} M & {b['ms_per_pair']['full_segmenter/b1']:.1f} & {b['ms_per_pair']['full_segmenter/b16']:.1f} \\\\",
    f"CoMIR encoders & {b['params']['comir'] / 1e6:.2f} M / {b['params']['comir'] / 1e6:.2f} M & -- & -- \\\\",
    f"Gated MI (one CPU thread) & -- & {b['ms_per_pair']['gated_mi/cpu1']:.1f} & -- \\\\",
    "\\bottomrule", "\\end{tabular}"]))

# Appendix B: DeepLIIF
T = {m: [nm, f"{N[f'dl/{m}/mis']:.2f}", "--" if m == "oracle" else f"{N[f'dl/{m}/ali']:.2f}"]
     for m, nm in (("none", "No alignment"), ("full_xa", "full\\_xa"), ("mi", "Gated MI"), ("cca_mean", "CCA (3 seeds)"), ("oracle", "Nominal oracle shift"))}
bold_best(T, [1, 2], ["full_xa", "mi", "cca_mean"], {1: "max", 2: "max"})
rows = [" & ".join(T[m]) + " \\\\" for m in ("none", "full_xa", "mi", "cca_mean", "oracle")]
write("tableB1_deepliif.tex", "\n".join(["\\begin{tabular}{@{}lcc@{}}", "\\toprule", "Method & Misaligned F1 (12) & Aligned F1 (8) \\\\",
                                          "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]))

# Appendix C: per-scene results
def sname(k):
    cond, rest = k.split("@")
    t, s = rest.split("/s")
    cond = dict(clean="clean", photon10="low photon", defocus4="defocus", all_photon10="low photon, both")[cond]
    return f"{'nuclear' if t == '0' else 'membrane'} & {cond} & {s}"


cols = ["none", "full_xa", "mi", "comir", "cca", "oracle"]
rows = []
for k in MIS + ALI:
    if k == ALI[0]:
        rows.append("\\midrule")
    vals = ["--" if (c == "oracle" and size(k) == 0) else f"{f1(c, k):.2f}" for c in cols]
    R = {i: [v] for i, v in enumerate(vals)}
    best = max(float(vals[i]) for i in (1, 2, 3, 4))
    vals = [f"\\textbf{{{v}}}" if i in (1, 2, 3, 4) and float(v) == best else v for i, v in enumerate(vals)]
    rows.append(f"{sname(k)} & " + " & ".join(vals) + " \\\\")
write("tableC1_scenes.tex", "\n".join(["\\begin{tabular}{@{}lllcccccc@{}}", "\\toprule",
                                        "Moved stream & Degradation & Shift (px) & None & full\\_xa & Gated MI & CoMIR & CCA & Oracle \\\\",
                                        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]))

# Appendix E: validation frame test
rows = []
fv = frame_val
rows.append(f"Clean (no shift) & {100 * fv['clean']['inst_f1']:.1f} & -- & {fv['clean']['count_rel']:.3f} & -- \\\\")
for t in (0, 1):
    for s in (4, 8, 16):
        r = fv[f"{t}/s{s}"]
        rows.append(f"{'Nuclear' if t == 0 else 'Membrane'} moved {s} px & {100 * r['fixed']['inst_f1']:.1f} & {100 * r['moved']['inst_f1']:.1f} & {r['fixed']['count_rel']:.3f} & {r['moved']['count_rel']:.3f} \\\\")
write("tableE1_frame_val.tex", "\n".join(["\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
                                          " & \\multicolumn{2}{c}{Instance F1} & \\multicolumn{2}{c}{Count relative error} \\\\",
                                          "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
                                          "Scene & Labels fixed & Labels moved & Labels fixed & Labels moved \\\\", "\\midrule",
                                          *rows, "\\bottomrule", "\\end{tabular}"]))

# Appendix F: per-marker real data
rows = []
for mk in MARKERS:
    r = N[f"real/m/{mk}"]
    bb = lambda x, y: (f"\\textbf{{{x:.1f}}}" if x >= y else f"{x:.1f}")
    a1, c1, a2, c2 = 100 * r['mi'], 100 * r['cca'], 100 * r['mi_within'], 100 * r['cca_within']
    a1, c1, a2, c2 = round(a1, 1), round(c1, 1), round(a2, 1), round(c2, 1)
    rows.append(f"{mk} & {r['n']} & {bb(a1, c1)} & {bb(c1, a1)} & {r['n_within']} & {bb(a2, c2)} & {bb(c2, a2)} \\\\")
write("tableF1_markers.tex", "\n".join(["\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
                                        " & \\multicolumn{3}{c}{Cross-cycle, residual $>$0} & \\multicolumn{3}{c}{Same-cycle} \\\\",
                                        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
                                        "Marker & Pairs & Gated MI & CCA & Pairs & Gated MI & CCA \\\\", "\\midrule",
                                        *rows, "\\bottomrule", "\\end{tabular}"]))


# ------------------------------------------------------------------ figures
def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


LABEL = dict(none="No alignment", full_mis="full_mis", unet_mis="unet_mis", full_xa="full_xa", mi="Gated MI",
             comir="CoMIR", cca="CCA (ours)", oracle="Oracle shift")
MARK = dict(none="o", full_mis="s", unet_mis="D", full_xa="^", mi="v", comir="P", cca="*")


def tradeoff(ax, big=False):
    fs = 11 if big else 7.5
    ax.axhline(N["oracle/mis"], ls="--", lw=0.8, color=C["oracle"])
    ax.text(7.9, N["oracle/mis"] + 0.6, f"Oracle shift ({N['oracle/mis']:.1f})", ha="right", va="bottom", fontsize=fs)
    off = dict(none=(0.25, -1.6), full_mis=(0.2, 1.6), unet_mis=(-0.25, 1.0), full_xa=(0.25, 0.9), mi=(-0.25, 1.1),
               comir=(0.3, 0.0), cca=(0.0, -3.0))
    ha = dict(none="left", full_mis="left", unet_mis="right", full_xa="left", mi="right", comir="left", cca="center")
    if big:
        off.update(cca=(0.45, -4.2), mi=(-0.3, -2.6), unet_mis=(0.2, -2.8), full_xa=(-0.2, 2.2))
        ha.update(cca="left", mi="right", unet_mis="right", full_xa="right")
    for m in ["none", "full_mis", "unet_mis", "full_xa", "mi", "comir", "cca"]:
        x, y = (0.0 if m == "none" else N[m + "/harm"]), N[m + "/mis"]
        ax.scatter(x, y, s=(260 if m == "cca" else 55) * (1.8 if big else 1), marker=MARK[m], color=C[m], edgecolor="k", linewidth=0.5, zorder=3)
        ax.annotate(LABEL[m], (x, y), xytext=(x + off[m][0], y + off[m][1]), fontsize=fs, ha=ha[m], va="center",
                    fontweight="bold" if m == "cca" else "normal")
    ax.set_xlim(-0.4, 8.2)
    ax.set_ylim(45, 82)
    ax.set_xlabel("Harm on aligned images (F1 points)" if big else "Harm on aligned images, $H$ (lower is better)", fontsize=fs)
    ax.set_ylabel("F1 on misaligned images" if big else "F1 on misaligned scenes (higher is better)", fontsize=fs)
    ax.tick_params(labelsize=fs)



# Fig. 3: frame-aware test
fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.5), sharey=True)
for t, ax in zip((0, 1), axes):
    xs = np.arange(3)
    fx = [100 * frame[f"{t}/s{s}"]["fixed"]["inst_f1"] for s in (4, 8, 16)]
    mv = [100 * frame[f"{t}/s{s}"]["moved"]["inst_f1"] for s in (4, 8, 16)]
    b1 = ax.bar(xs - 0.19, fx, 0.36, color="#C7CCD3", edgecolor="white", lw=0.6, label="Labels in the reference frame")
    b2 = ax.bar(xs + 0.19, mv, 0.36, color=C["cca"], edgecolor="white", lw=0.6, label="Labels moved with the shifted channel")
    ax.axhline(100 * frame["clean"]["inst_f1"], ls=(0, (4, 2)), color="0.3", lw=0.8)
    ax.text(2.55, 100 * frame["clean"]["inst_f1"] + 1.5, f"no shift {100 * frame['clean']['inst_f1']:.1f}", ha="right", fontsize=7.5)
    for bars, dark in ((b1, False), (b2, True)):
        for r in bars:
            h = r.get_height()
            inside = h > 30
            ax.text(r.get_x() + r.get_width() / 2, h - 2 if inside else h + 1.2, f"{h:.1f}", ha="center",
                    va="top" if inside else "bottom", fontsize=7, color=("white" if dark else "0.15") if inside else "0.15")
    ax.set_xticks(xs, [f"{s} px" for s in (4, 8, 16)])
    ax.set_title(("(a) Nuclear channel moved" if t == 0 else "(b) Membrane channel moved"), fontsize=8.5, loc="left")
    ax.set_ylim(0, 100)
axes[0].set_ylabel("Instance F1 (IoU 0.5)")
axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.15), ncol=2)
axes[0].set_xlabel("")
save(fig, "fig3_frame_test")

# Fig. 4: operating range and failure quartiles
fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
ax = axes[0]
for m in ("cca", "comir", "mi"):
    ax.plot(SIZES, [100 * v for v in N[f"range/{m}"]], marker=MARK[m], ms=6 if m == "cca" else 5, color=C[m], lw=1.2,
            label={"cca": "CCA ($\\pm$36 px)", "comir": "CoMIR ($\\pm$24 px)", "mi": "Gated MI ($\\pm$24 px)"}[m])
ax.axvspan(24, 41, color="0.94", zorder=0, lw=0)
ax.text(34.3, 98, "outside the\n$\\pm$24 px search\nof CoMIR and MI", ha="center", va="top", fontsize=7, color="0.3", linespacing=1.15)
ax.set_xlabel("Applied shift per axis (px)")
ax.set_ylabel("Shift error ≤2 px (%)")
ax.set_xticks(SIZES)
ax.set_ylim(-3, 100)
ax.set_xlim(2, 41)
ax.legend(loc="lower left", fontsize=7)
ax.set_title("(a) Operating range", fontsize=8.5, loc="left")
ax = axes[1]
q_cca = 100 * np.mean([N["fail/t0"]["q_cca"], N["fail/t1"]["q_cca"]], 0)
q_com = 100 * np.mean([N["fail/t0"]["q_comir"], N["fail/t1"]["q_comir"]], 0)
xs = np.arange(4)
for dx, q, key, name in ((-0.19, q_cca, "cca", "CCA"), (0.19, q_com, "comir", "CoMIR")):
    bars = ax.bar(xs + dx, q, 0.36, color=C[key], edgecolor="white", lw=0.6, label=name)
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.6, f"{r.get_height():.1f}", ha="center", va="bottom", fontsize=6.5, color="0.15")
ax.set_ylim(0, 46)
ax.set_xticks(xs, ["Q1\n(fewest)", "Q2", "Q3", "Q4\n(most)"])
ax.set_xlabel("Cell-count quartile")
ax.set_ylabel("Failure rate, error >2 px (%)")
ax.legend(fontsize=7, frameon=True, facecolor="white", edgecolor="none", framealpha=1)
ax.set_title("(b) Failures at a 16 px shift", fontsize=8.5, loc="left")
fig.tight_layout()
save(fig, "fig4_range_failure")

# Fig. 5: qualitative examples (linear percentile stretch only)
from skimage.segmentation import find_boundaries

arr = np.load(os.path.join(RES, "qual_tissuenet.npz"))
picks = {p["tag"]: p for p in qual["picks"]}


def composite(streams):
    def norm(v):
        lo, hi = np.percentile(v, 1), np.percentile(v, 99.5)
        return np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
    n, m = norm(streams[0]), norm(streams[1])
    return np.stack([n, m, n], -1)


def overlay(streams, pred, label):
    img = composite(streams) * 0.85
    img[find_boundaries(label, mode="inner")] = (1.0, 1.0, 1.0)
    img[find_boundaries(pred, mode="inner")] = (1.0, 0.65, 0.0)
    return img


ROWS = [("p1", "Nuclear moved 16 px\n(median CCA image)"), ("p4", "Membrane moved 16 px\n(median CCA image)"),
        ("p7", "Already aligned\n(median MI harm)"), ("p6", "Nuclear moved 16 px\n(CCA failure)")]
COLS = [("input", "Input channels"), ("none", "No alignment"), ("mi", "Gated MI"), ("cca", "CCA"), ("oracle", "Oracle shift")]
fig, axes = plt.subplots(len(ROWS), len(COLS), figsize=(5.4, 4.75))
for r, (tag, rowname) in enumerate(ROWS):
    p = picks[tag]
    for c, (key, colname) in enumerate(COLS):
        ax = axes[r, c]
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if r == 0:
            ax.set_title(colname, fontsize=8)
        if key == "input":
            ax.imshow(composite(arr[f"{tag}_input"].astype(np.float32)))
            ax.set_ylabel(rowname, fontsize=7.5)
            ax.text(3, 218, f"{p['cells']} cells", color="w", fontsize=6.5, bbox=dict(fc="k", alpha=0.65, pad=1.2, lw=0))
            continue
        if f"{tag}_{key}_pred" not in arr:
            ax.text(0.5, 0.5, "identical to\nno alignment", ha="center", va="center", fontsize=7, transform=ax.transAxes, color="0.4")
            continue
        ax.imshow(overlay(arr[f"{tag}_{key}_streams"].astype(np.float32), arr[f"{tag}_{key}_pred"], arr[f"{tag}_label"]))
        mth = p["methods"][key]
        ax.text(3, 14, f"F1 {mth['inst_f1']:.1f}", color="w", fontsize=7, fontweight="bold", bbox=dict(fc="k", alpha=0.65, pad=1.2, lw=0))
        ax.text(3, 218, f"shift ({mth['shift'][0]}, {mth['shift'][1]})", color="w", fontsize=6.5, bbox=dict(fc="k", alpha=0.65, pad=1.2, lw=0))
fig.subplots_adjust(wspace=0.03, hspace=0.06)
save(fig, "fig5_qualitative")


# Fig. 2: method diagram (inch coordinates at the 5.4 in text width; thumbnails from TissueNet test image p1)
from matplotlib.patches import Circle, Polygon, Rectangle

FROZ, TRAIN, OP, OUTC = "#DCE6F2", "#FBE2C0", "#F0F0F0", "#DCEFE4"
npz = arr


def thumb(ax, img, x, y, w):
    ax.imshow(img, extent=(x, x + w, y, y + w), interpolation="lanczos", zorder=2)
    ax.add_patch(Rectangle((x, y), w, w, fill=False, ec="0.3", lw=0.6, zorder=3))


def tint(v, rgb):
    lo, hi = np.percentile(v, 1), np.percentile(v, 99.5)
    v = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.stack([v * rgb[0], v * rgb[1], v * rgb[2]], -1)


def grid_icon(ax, x, y, w, n, fc, peak=None, sigma=1.1):
    c = w / n
    for i in range(n):
        for j in range(n):
            col = fc
            if peak is not None:
                val = np.exp(-((i - peak[0]) ** 2 + (j - peak[1]) ** 2) / (2 * sigma ** 2))
                col = plt.cm.Blues(0.08 + 0.85 * val)
            ax.add_patch(Rectangle((x + j * c, y + (n - 1 - i) * c), c, c, fc=col, ec="white", lw=0.35, zorder=2))
    ax.add_patch(Rectangle((x, y), w, w, fill=False, ec="0.35", lw=0.6, zorder=3))


def rbox(ax, x, y, w, h, text, fc, fs=7, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.05", fc=fc, ec="0.3", lw=0.6, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", zorder=4)


def go(ax, a, b, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, lw=0.75, color="0.15", ls=ls, shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))


def op(ax, x, y, sym, r=0.1):
    ax.add_patch(Circle((x, y), r, fc="white", ec="0.2", lw=0.7, zorder=3))
    ax.text(x, y, sym, ha="center", va="center", fontsize=9, zorder=4)


def cnn_icon(ax, x, y, h, fc):
    for k in range(3):
        ax.add_patch(Polygon([(x + 0.13 * k, y + 0.06 * k), (x + 0.13 * k + 0.08, y + 0.06 * k + 0.06),
                              (x + 0.13 * k + 0.08, y + h - 0.06 * k + 0.06), (x + 0.13 * k, y + h - 0.06 * k)],
                             closed=True, fc=fc, ec="0.3", lw=0.6, zorder=2))


ref_img = tint(npz["p1_input"][1].astype(np.float32), (0.3, 1.0, 0.3))
mov_img = tint(npz["p1_input"][0].astype(np.float32), (1.0, 0.25, 1.0))
warp_img = tint(npz["p1_cca_streams"][0].astype(np.float32), (1.0, 0.25, 1.0))
out_img = composite(npz["p1_cca_streams"].astype(np.float32))

fig, ax = plt.subplots(figsize=(5.4, 2.95))
fig.subplots_adjust(0, 0, 1, 1)
ax.set_xlim(0, 5.4)
ax.set_ylim(0, 2.95)
ax.set_aspect("equal")
ax.axis("off")
ax.grid(False)
# coarse level (top row)
ax.text(0.02, 2.9, "Coarse level", fontsize=8, fontweight="bold", va="top")
thumb(ax, ref_img, 0.05, 1.98, 0.56)
thumb(ax, mov_img, 0.05, 1.28, 0.56)
ax.text(0.33, 2.57, "reference $x^r$", ha="center", va="bottom", fontsize=7)
ax.text(0.33, 1.23, "moving $x^m$", ha="center", va="top", fontsize=7)
ax.add_patch(Polygon([(0.82, 1.28), (0.82, 2.54), (1.45, 2.34), (1.45, 1.48)], closed=True, fc=FROZ, ec="0.3", lw=0.6, zorder=2))
ax.text(1.135, 1.91, "DINOv2\nViT-B/14\nfrozen,\nshared", ha="center", va="center", fontsize=7, zorder=4, linespacing=1.15)
go(ax, (0.61, 2.26), (0.82, 2.26))
go(ax, (0.61, 1.56), (0.82, 1.56))
grid_icon(ax, 1.62, 2.08, 0.36, 4, FROZ)
grid_icon(ax, 1.62, 1.38, 0.36, 4, FROZ)
ax.text(1.8, 1.91, "tokens\n(7 px)", ha="center", va="center", fontsize=6.5, color="0.25")
go(ax, (1.45, 2.26), (1.62, 2.26))
go(ax, (1.45, 1.56), (1.62, 1.56))
rbox(ax, 2.14, 2.12, 0.46, 0.28, "MLP$^r$", TRAIN)
rbox(ax, 2.14, 1.42, 0.46, 0.28, "MLP$^m$", TRAIN)
go(ax, (1.98, 2.26), (2.14, 2.26))
go(ax, (1.98, 1.56), (2.14, 1.56))
op(ax, 2.86, 1.91, r"$\otimes$")
go(ax, (2.6, 2.26), (2.79, 1.98))
go(ax, (2.6, 1.56), (2.79, 1.84))
grid_icon(ax, 3.1, 1.64, 0.54, 9, OP, peak=(2.2, 6.0))
ax.text(3.37, 2.23, "coarse cost\n$\\pm$4 tokens ($\\pm$28 px)", ha="center", va="bottom", fontsize=6.5)
go(ax, (2.96, 1.91), (3.1, 1.91))
go(ax, (3.64, 1.91), (4.2, 1.91))
ax.text(3.92, 1.96, "soft-argmax", ha="center", va="bottom", fontsize=6.5)
ax.text(4.3, 1.91, "$d_0$", ha="center", va="center", fontsize=9)
# fine level (bottom row)
ax.text(1.12, 1.16, "Fine level", fontsize=8, fontweight="bold", va="top")
go(ax, (0.33, 1.13), (0.33, 0.92))
ax.text(0.4, 1.03, "warp by $d_0$", ha="left", va="center", fontsize=6.5)
thumb(ax, warp_img, 0.05, 0.34, 0.56)
cnn_icon(ax, 0.86, 0.38, 0.42, TRAIN)
ax.text(1.06, 0.31, "per-channel CNN, 2 px", ha="center", va="top", fontsize=6.5)
go(ax, (0.61, 0.62), (0.84, 0.62))
op(ax, 1.62, 0.62, r"$\otimes$")
go(ax, (1.3, 0.62), (1.52, 0.62))
ax.text(1.62, 0.78, "with $x^r$", ha="center", va="bottom", fontsize=6.5, color="0.25")
grid_icon(ax, 1.86, 0.35, 0.54, 9, OP, peak=(4.6, 3.6))
ax.text(2.13, 0.94, "fine cost\n$\\pm$4 cells ($\\pm$8 px)", ha="center", va="bottom", fontsize=6.5)
go(ax, (1.72, 0.62), (1.86, 0.62))
go(ax, (2.4, 0.62), (2.92, 0.62))
ax.text(2.66, 0.67, "soft-argmax", ha="center", va="bottom", fontsize=6.5)
ax.text(3.02, 0.62, "$d_1$", ha="center", va="center", fontsize=9)
op(ax, 3.4, 0.62, "+")
go(ax, (3.12, 0.62), (3.3, 0.62))
go(ax, (4.3, 1.8), (3.46, 0.71), rad=-0.15)
rbox(ax, 3.63, 0.38, 0.95, 0.48, "median of four\nrotations, round,\n$\\pm$1 px dead zone", OP, fs=6.5)
go(ax, (3.5, 0.62), (3.63, 0.62))
thumb(ax, out_img, 4.78, 0.34, 0.56)
go(ax, (4.58, 0.62), (4.78, 0.62))
ax.text(5.06, 0.3, "aligned pair $\\to$\nany segmenter", ha="center", va="top", fontsize=6.5, fontweight="bold")
# training note and legend
ax.add_patch(FancyBboxPatch((4.42, 1.5), 0.94, 1.32, boxstyle="round,pad=0.015,rounding_size=0.05", fc="white", ec="#B85C00", lw=0.7,
                            ls=(0, (3, 2)), zorder=2))
ax.text(4.89, 2.16, "Training\n(no cell labels)\n\nrandom shift of\none channel\n+ physics\ndegradation\n\nsmooth-L1 on $d$, $d_0$\n+ CE on coarse cell",
        ha="center", va="center", fontsize=6.3, color="#8A4500", zorder=4, linespacing=1.12)
for i, (fc, t) in enumerate([(FROZ, "frozen"), (TRAIN, "trainable (0.55 M)"), (OP, "parameter-free")]):
    x0 = [0.06, 0.78, 1.82][i]
    ax.add_patch(Rectangle((x0, 0.04), 0.14, 0.1, fc=fc, ec="0.3", lw=0.5))
    ax.text(x0 + 0.19, 0.09, t, fontsize=6.5, va="center")
save(fig, "fig2_architecture")

# Fig. 1: teaser. (a) median-CCA example (nuclear moved 16 px, rule fixed before rendering); (b) trade-off
fig = plt.figure(figsize=(5.4, 2.62))
W, H = 5.4, 2.62
sz = 1.0
pos = [(0.08, 1.33), (1.16, 1.33), (0.08, 0.13), (1.16, 0.13)]
p1 = picks["p1"]
panels = [("input", "Input channels\n(nuclear moved 16 px)"), ("none", f"No alignment\nF1 {p1['methods']['none']['inst_f1']:.1f}"),
          ("mi", f"Gated MI\nF1 {p1['methods']['mi']['inst_f1']:.1f}"), ("cca", f"CCA\nF1 {p1['methods']['cca']['inst_f1']:.1f}")]
for (x0, y0), (key, title) in zip(pos, panels):
    ax = fig.add_axes([x0 / W, y0 / H, sz / W, sz / H])
    if key == "input":
        ax.imshow(composite(arr["p1_input"].astype(np.float32)))
    else:
        ax.imshow(overlay(arr[f"p1_{key}_streams"].astype(np.float32), arr[f"p1_{key}_pred"], arr["p1_label"]))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("0.4")
        sp.set_linewidth(0.5)
    ax.set_title(title, fontsize=7, pad=2, linespacing=1.1, fontweight="bold" if key == "cca" else "normal")
fig.text(0.0, 0.985, "(a)", fontsize=8.5, fontweight="bold", va="top")
fig.text(2.45 / W, 0.985, "(b)", fontsize=8.5, fontweight="bold", va="top")
ax = fig.add_axes([2.95 / W, 0.42 / H, 2.38 / W, 2.05 / H])
tradeoff(ax)
save(fig, "fig1_tradeoff")

# Graphical abstract: >= 1328 px wide, landscape, readable at 13 cm
fig = plt.figure(figsize=(13 / 2.54 * 1.6, 5.5 / 2.54 * 1.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.45], wspace=0.5)
p = picks["p1"]
for i, (key, title) in enumerate((("none", "Misaligned channels"), ("cca", "After CCA"))):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(overlay(arr[f"p1_{key}_streams"].astype(np.float32), arr[f"p1_{key}_pred"], arr["p1_label"]))
    ax.set_title(f"{title}\nF1 {p['methods'][key]['inst_f1']:.1f}", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
ax = fig.add_subplot(gs[0, 2])
tradeoff(ax, big=True)
fig.suptitle("Align before you segment: label-free cross-channel alignment, plug-in to any segmenter", fontsize=12, y=1.04)
fig.savefig(os.path.join(FIG, "graphical_abstract.png"), bbox_inches="tight", dpi=300)
fig.savefig(os.path.join(FIG, "graphical_abstract.pdf"), bbox_inches="tight", dpi=300)
plt.close(fig)

from PIL import Image

w, h = Image.open(os.path.join(FIG, "graphical_abstract.png")).size
print("graphical abstract px", w, h)
print("key numbers:", {k: round(N[k], 2) for k in ["none/mis", "oracle/mis", "cca/mis", "cca/G", "cca/harm", "comir/G", "comir/harm", "mi/G", "mi/harm", "full_xa/G", "cca/G_nuc", "cca/G_mem", "comir/G_nuc", "comir/G_mem", "gap", "gap_nuc", "gap_mem"]})
