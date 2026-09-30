"""
03_fault_library.py
Does each fault leave a DISTINCT metric signature?

Faults (3 severities each):
  ring        detector channel gain error
  hu_drift    water calibration shift
  blur        resolution loss (focal spot growth)
  cupping     beam-hardening correction error
  tube_decl   tube output decline (fewer photons)

Metrics (all in HU unless stated):
  mean_hu     centre ROI mean      (CT number accuracy)
  noise_hu    centre ROI SD        (noise)
  unif_hu     periphery - centre   (uniformity, signed)
  ring_hu     radial-profile bump  (ring detector)
  edge_px     10-90% edge width    (resolution proxy, pixels)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import ctsim

rng = np.random.default_rng(seed=11)
phantom = ctsim.make_phantom(with_inserts=False)
sino = ctsim.forward(phantom)
n_det = sino.shape[0]
det_c = n_det // 2
I0_BASE = 1e5
C = ctsim.CENTER


# ------------------------------------------------------------ metrics
def _roi(img, x, y, half=12):
    return img[y - half:y + half, x - half:x + half]


def radial_profile(img, r_max=130):
    yy, xx = np.mgrid[:ctsim.N, :ctsim.N]
    r = np.hypot(xx - C, yy - C).astype(int)
    return np.array([img[r == k].mean() for k in range(r_max)])


def edge_width(prof):
    """10-90% width of the phantom edge (true edge at r=100)."""
    plateau = prof[80:92].mean()
    hi, lo = 0.9 * plateau, 0.1 * plateau
    r = np.arange(len(prof))
    seg = slice(92, 125)
    p, rr = prof[seg], r[seg]
    # profile falls with radius: find crossings by linear interpolation
    def cross(level):
        idx = np.where(p < level)[0][0]
        x0, x1, y0, y1 = rr[idx - 1], rr[idx], p[idx - 1], p[idx]
        return x0 + (y0 - level) / (y0 - y1) * (x1 - x0)
    return cross(lo) - cross(hi)


def measure(img_mu):
    hu = ctsim.to_hu(img_mu)
    centre = _roi(hu, C, C)
    periph = np.mean([_roi(hu, C + 70, C).mean(), _roi(hu, C - 70, C).mean(),
                      _roi(hu, C, C + 70).mean(), _roi(hu, C, C - 70).mean()])
    prof_hu = radial_profile(hu, 90)
    trend = median_filter(prof_hu, size=21, mode="nearest")
    ring = np.abs((prof_hu - trend)[5:]).max()
    return {
        "mean_hu": centre.mean(),
        "noise_hu": centre.std(),
        "unif_hu": periph - centre.mean(),
        "ring_hu": ring,
        "edge_px": edge_width(radial_profile(img_mu)),
    }


# ------------------------------------------------------------ faults
def gain_vec(err, offset=40):
    g = np.ones(n_det)
    g[det_c + offset] *= 1 + err
    return g


FAULTS = {
    "ring":      [("1%", dict(channel_gain=gain_vec(0.01))),
                  ("2%", dict(channel_gain=gain_vec(0.02))),
                  ("5%", dict(channel_gain=gain_vec(0.05)))],
    "hu_drift":  [("+2 HU", dict(hu_shift=2)),
                  ("+5 HU", dict(hu_shift=5)),
                  ("+10 HU", dict(hu_shift=10))],
    "blur":      [("s=0.5", dict(blur_sigma=0.5)),
                  ("s=1.0", dict(blur_sigma=1.0)),
                  ("s=2.0", dict(blur_sigma=2.0))],
    "cupping":   [("k=0.01", dict(bh_k=0.01)),
                  ("k=0.03", dict(bh_k=0.03)),
                  ("k=0.06", dict(bh_k=0.06))],
    "tube_decl": [("70%", dict(_I0=0.7 * I0_BASE)),
                  ("40%", dict(_I0=0.4 * I0_BASE)),
                  ("20%", dict(_I0=0.2 * I0_BASE))],
}
METRICS = ["mean_hu", "noise_hu", "unif_hu", "ring_hu", "edge_px"]


def run(kwargs):
    kw = dict(kwargs)
    I0 = kw.pop("_I0", I0_BASE)
    return measure(ctsim.simulate_scan(sino, I0, rng, **kw))


# ------------------------------------------------------------ baseline
N_BASE, N_REP = 12, 5
base = [measure(ctsim.simulate_scan(sino, I0_BASE, rng)) for _ in range(N_BASE)]
b_mean = {m: np.mean([b[m] for b in base]) for m in METRICS}
b_std = {m: np.std([b[m] for b in base], ddof=1) for m in METRICS}
print("BASELINE (no fault), mean +/- SD over", N_BASE, "scans")
for m in METRICS:
    print(f"  {m:9s} {b_mean[m]:9.3f} +/- {b_std[m]:.3f}")

# ------------------------------------------------------------ sweep
rows, labels, vals = [], [], []
print(f"\n{'fault':10s}{'level':8s}" + "".join(f"{m:>10s}" for m in METRICS))
for fault, levels in FAULTS.items():
    for name, kw in levels:
        reps = [run(kw) for _ in range(N_REP)]
        mean = {m: np.mean([r[m] for r in reps]) for m in METRICS}
        z = [(mean[m] - b_mean[m]) / b_std[m] for m in METRICS]
        labels.append(f"{fault} {name}")
        vals.append(z)
        print(f"{fault:10s}{name:8s}" + "".join(f"{mean[m]:10.3f}" for m in METRICS))
vals = np.array(vals)

# ------------------------------------------------------------ signature heatmap
# Signed, log-compressed z-score: distance from baseline in units of the
# baseline scan-to-scan SD.
disp = np.sign(vals) * np.log10(1 + np.abs(vals))
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(disp, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
ax.set_xticks(range(len(METRICS)))
ax.set_xticklabels(METRICS)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
for i in range(vals.shape[0]):
    for j in range(vals.shape[1]):
        ax.text(j, i, f"{vals[i, j]:.0f}", ha="center", va="center", fontsize=7)
ax.set_title("Fault signatures: deviation from baseline (z-score, numbers shown)")
plt.colorbar(im, label="signed log10(1+|z|)")
plt.tight_layout()
plt.savefig("fault_signatures.png", dpi=150)
print("\nSaved fault_signatures.png")