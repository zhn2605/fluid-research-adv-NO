# URTC poster figures: rolls each saved model out 30 steps on the test split
# usage: .venv/bin/python make_poster_figures.py  -> figures/poster/
import glob
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

RUNS = {  # Re -> run per model
    240:  {"Baseline": "UNet/20260901-122104", "PITA": "UNet+PITA/20260901-122112", "adv-NO": "UNet+advNO/20260901-163527"},
    1280: {"Baseline": "UNet/20260815-210448", "PITA": "UNet+PITA/20260815-210457", "adv-NO": "UNet+advNO/20260815-210506"},
    2520: {"Baseline": "UNet/20260907-211749", "PITA": "UNet+PITA/20260907-211757", "adv-NO": "UNet+advNO/20260907-211805"},
}
# colorblind-safe colors
STYLE = {
    "Baseline": dict(color="#0072b2", ls="--", marker="o"),
    "PITA":     dict(color="#d55e00", ls=":",  marker="s"),
    "adv-NO":   dict(color="#009e73", ls="-",  marker="^"),
}
T, SEED, NTEST, NBOOT = 30, 5496, 48, 2000
HIGH_K = slice(-10, None)  # last 10 bins = fine scales (k >= 0.375)
OUT = Path("figures/poster")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(0)

plt.rcParams.update({
    "font.size": 28, "axes.labelsize": 30, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.5,
    "grid.color": "#d9d9d9", "grid.linewidth": 1, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def spectrum(f, nbins=40):
    # radially binned energy spectrum of one [H, W] field, DC bin dropped
    H, W = f.shape
    P = np.abs(np.fft.fft2(f)) ** 2 / (H * W)
    k = np.sqrt(np.add.outer(np.fft.fftfreq(H) ** 2, np.fft.fftfreq(W) ** 2))
    b = np.linspace(0, 0.5, nbins + 1)
    E = np.histogram(k, b, weights=P)[0] / np.maximum(np.histogram(k, b)[0], 1)
    return 0.5 * (b[1:] + b[:-1])[1:], E[1:]


def load(re):
    # test data and each model's 30-step rollout in m/s, [C, B, H, W, T+1]
    arms = RUNS[re]
    with h5py.File(glob.glob(f"outputs/{arms['Baseline']}*.h5")[0]) as f:
        st = {k: float(f[k][()]) for k in ("u_x_mean", "u_x_std", "u_y_mean", "u_y_std")}
    with h5py.File(f".data/Re_{re}.h5") as f:
        state = np.random.get_state(); np.random.seed(SEED)
        p = np.random.permutation(f["u_x"].shape[0]); np.random.set_state(state)
        idx = p[-NTEST:]; order = np.argsort(idx)
        ux = np.empty((NTEST, T + 1, 89, 133)); uy = np.empty_like(ux)
        ux[order] = f["u_x"][np.sort(idx), : T + 1]; uy[order] = f["u_y"][np.sort(idx), : T + 1]
    act = np.stack([ux, uy]).transpose(0, 1, 4, 3, 2).astype(np.float32)  # same axis order as MatReader
    mean, std = [st["u_x_mean"], st["u_y_mean"]], [st["u_x_std"], st["u_y_std"]]
    u0 = torch.from_numpy(np.stack([(act[c, ..., 0] - mean[c]) / std[c] for c in (0, 1)], 1)).to(dev)
    preds = {}
    for arm, stem in arms.items():
        pt = glob.glob(f"outputs/{stem}*.pt")[0]
        model = torch.load(pt, map_location=dev, weights_only=False).eval()
        seq, u = [u0], u0
        with torch.no_grad():
            for _ in range(T):
                u = model(u); seq.append(u)
        pr = torch.stack(seq, 1).permute(2, 0, 3, 4, 1).cpu().numpy()
        for c in (0, 1):
            pr[c] = pr[c] * std[c] + mean[c]
        with h5py.File(pt[:-3] + ".h5") as f:  # check against the saved rollout
            sv = f["u_pred"][:]
        for c in (0, 1):
            sv[c] = sv[c] * std[c] + mean[c]
        assert np.abs(pr[..., :11] - sv).max() < 1e-2, (re, arm)
        preds[arm] = pr
        del model; torch.cuda.empty_cache()
    return act, preds


def per_seq_spectra(arr, t, nbins=40):
    # [B, nbins] spectrum per sequence, u + v
    return np.array([spectrum(arr[0, s, :, :, t], nbins)[1] + spectrum(arr[1, s, :, :, t], nbins)[1] for s in range(arr.shape[1])])


def boot(stat, n=NTEST):
    # 95% bootstrap interval over test sequences
    draws = np.array([stat(rng.integers(0, n, n)) for _ in range(NBOOT)])
    return np.percentile(draws, [2.5, 97.5], axis=0)


# compute
data, k = {}, spectrum(np.zeros((133, 89)))[0]
PLOT_BINS = 20  # coarser bins for Fig 2 plot only
k_plot = spectrum(np.zeros((133, 89)), PLOT_BINS)[0]
for re in RUNS:
    print(f"rolling out Re {re}")
    act, preds = load(re)
    Sa = per_seq_spectra(act, T)
    Sa_plot = per_seq_spectra(act, T, PLOT_BINS)
    d = {"act": act if re == 2520 else None, "preds": preds if re == 2520 else None, "ratio": {}, "ratio_plot": {}, "l2": {}, "l2_curve": {}, "hk_seq": {}, "l2_seq": {}}
    for arm, pr in preds.items():
        Sp = per_seq_spectra(pr, T)
        r = lambda i: Sp[i].mean(0) / Sa[i].mean(0)  # energy kept per scale
        d["ratio"][arm] = (r(np.arange(NTEST)), boot(r))
        Sp_plot = per_seq_spectra(pr, T, PLOT_BINS)
        rp = lambda i: Sp_plot[i].mean(0) / Sa_plot[i].mean(0)
        d["ratio_plot"][arm] = (rp(np.arange(NTEST)), boot(rp))
        err, ref = act[..., T] - pr[..., T], act[..., T]
        l2 = lambda i: np.linalg.norm(err[:, i]) / np.linalg.norm(ref[:, i])
        d["l2"][arm] = (l2(np.arange(NTEST)), boot(l2))
        d["hk_seq"][arm] = Sp[:, HIGH_K].sum(1) / Sa[:, HIGH_K].sum(1)
        d["l2_seq"][arm] = np.array([l2([i]) for i in range(NTEST)])
        e2 = ((act - pr) ** 2).sum(axis=(0, 2, 3))[:, 1:]  # [B, T]
        r2 = (act ** 2).sum(axis=(0, 2, 3))[:, 1:]
        curve = lambda i: np.sqrt(e2[i].sum(0) / r2[i].sum(0))
        d["l2_curve"][arm] = (curve(np.arange(NTEST)), boot(curve))
    data[re] = d

# stats
print("\nstep 30, paired over 48 test sequences (Wilcoxon signed-rank)")
for re, d in data.items():
    for arm in ("PITA", "adv-NO"):
        hk, hb = d["hk_seq"][arm], d["hk_seq"]["Baseline"]
        la, lb = d["l2_seq"][arm], d["l2_seq"]["Baseline"]
        print(f"Re {re:>4} {arm:<7} fine-scale kept: {np.mean(d['ratio'][arm][0][HIGH_K]):.2f} vs {np.mean(d['ratio']['Baseline'][0][HIGH_K]):.2f}, "
              f"more in {(hk > hb).sum()}/48 (p={wilcoxon(hk, hb).pvalue:.1e}) | "
              f"L2 {d['l2'][arm][0]:.4f} vs {d['l2']['Baseline'][0]:.4f} ({d['l2'][arm][0] / d['l2']['Baseline'][0] - 1:+.0%}), "
              f"higher in {(la > lb).sum()}/48 (p={wilcoxon(la, lb).pvalue:.1e})")

OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)


# Fig 1: rollout at Re 2520
STEPS = [0, 5, 10, 15, 20, 25, 30]


def field_grid(rows, steps, cmap, norm, cbar_label, name):
    # rows x steps grid of fields with one colorbar
    fig, axes = plt.subplots(len(rows), len(steps), figsize=(21, 1.75 * len(rows) + 0.8), squeeze=False)
    fig.subplots_adjust(left=0.13, right=0.91, top=1 - 0.7 / (1.75 * len(rows) + 0.8), bottom=0.02, wspace=0.05, hspace=0.08)
    for r, (label, arr, color) in enumerate(rows):
        for c, t in enumerate(steps):
            ax = axes[r, c]
            im = ax.imshow(arr[0, 0, :, :, t].T, origin="lower", cmap=cmap, norm=norm, interpolation="none")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(f"step {t}", fontsize=28, pad=10)
        axes[r, 0].set_ylabel(label, rotation=0, ha="right", va="center", labelpad=32, fontsize=28)
        axes[r, 0].plot([-0.06, -0.06], [0.08, 0.92], color=color, lw=8, transform=axes[r, 0].transAxes,
                        clip_on=False, solid_capstyle="round")
    top, bot = axes[0, -1].get_position(), axes[-1, -1].get_position()
    cax = fig.add_axes([0.925, bot.y0, 0.012, top.y1 - bot.y0])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(cbar_label, fontsize=28); cb.outline.set_visible(False)
    save(fig, name)
    return cb


act, preds = data[2520]["act"], data[2520]["preds"]
rows = [("Measured", act, "#1a1a1a")] + [(arm, preds[arm], STYLE[arm]["color"]) for arm in STYLE]
field_grid(rows, STEPS, "RdBu_r", TwoSlopeNorm(vmin=-0.15, vcenter=0, vmax=0.40),
           "velocity (m/s)", "Fig1_Rollout_Re2520")

# Fig 2: energy kept at each scale
fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharey=True)
for ax, (re, d) in zip(axes, data.items()):
    ax.axvspan(0.375, 0.5, color="#eeeeee", zorder=0)
    ax.axhline(1, color="#7f7f7f", lw=2, zorder=1)
    for arm, st in STYLE.items():
        mid, (lo, hi) = d["ratio_plot"][arm]
        ax.fill_between(k_plot, lo, hi, color=st["color"], alpha=0.2, lw=0, zorder=2)
        ax.plot(k_plot, mid, color=st["color"], ls=st["ls"], lw=4, marker=st["marker"], ms=13, markevery=2, label=arm, zorder=3)
    ax.set_title(f"Re {re}")
    ax.set_xlim(0, 0.5); ax.set_ylim(0, 1.15)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], ["0", ".1", ".2", ".3", ".4", ".5"])
    ax.grid(True, axis="y")
axes[0].set_ylabel("energy kept")
axes[1].set_xlabel("wavenumber k  (larger = finer scales)")
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.55, 1.08), handlelength=2.5)
fig.tight_layout()
save(fig, "Fig2_EnergyKept_perRe")

# Fig 3: error growth over the rollout
steps = np.arange(1, T + 1)
fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharey=True)
for ax, (re, d) in zip(axes, data.items()):
    ax.axvline(10, color="#7f7f7f", lw=2, zorder=1)
    for arm, st in STYLE.items():
        mid, (lo, hi) = d["l2_curve"][arm]
        ax.fill_between(steps, lo, hi, color=st["color"], alpha=0.2, lw=0, zorder=2)
        ax.plot(steps, mid, color=st["color"], ls=st["ls"], lw=4, marker=st["marker"], ms=13, markevery=[0, 9, 19, 29], label=arm, zorder=3)
    adv, base = d["l2_curve"]["adv-NO"][0][-1], d["l2_curve"]["Baseline"][0][-1]
    ax.text(30, adv + 0.007, f"{adv / base - 1:+.0%}", ha="right", va="bottom", fontsize=28)
    ax.set_title(f"Re {re}")
    ax.set_xlim(0, 31); ax.set_ylim(0, 0.17)
    ax.set_xticks([1, 10, 20, 30])
    ax.grid(True, axis="y")
axes[0].text(10.8, 0.158, "trained on\n10 steps", ha="left", va="top", fontsize=28, color="#555555")
axes[0].set_ylabel("relative L2 error")
axes[1].set_xlabel("forecast step")
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.55, 1.08), handlelength=2.5)
fig.tight_layout()
save(fig, "Fig3_ErrorGrowth_perRe")
print(f"\nsaved -> {OUT}/")
