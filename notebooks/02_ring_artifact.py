"""
02_ring_artifact.py
Inject a detector-channel gain error (ring artifact) and ask:
  1. Does the ring appear at the physically correct radius?
  2. Does the classic uniformity metric notice it?
  3. Does a radial-profile metric notice it?
  4. How does detection depend on fault severity and dose?
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import ctsim

rng = np.random.default_rng(seed=7)
phantom = ctsim.make_phantom(with_inserts=False)   # clean water cylinder for QA
sino = ctsim.forward(phantom)
n_det = sino.shape[0]
det_center = n_det // 2


# ------------------------------------------------------------ metrics
def uniformity(img, r_periph=70, half=12):
    """ACR-style: centre ROI mean vs mean of 4 peripheral ROI means.
    Returns the largest absolute difference (in attenuation units)."""
    c = ctsim.CENTER

    def roi_mean(x, y):
        return img[y - half:y + half, x - half:x + half].mean()

    centre = roi_mean(c, c)
    periph = [roi_mean(c + r_periph, c), roi_mean(c - r_periph, c),
              roi_mean(c, c + r_periph), roi_mean(c, c - r_periph)]
    return max(abs(p - centre) for p in periph)


def radial_profile(img, r_max=90):
    """Mean pixel value as a function of distance from the centre."""
    yy, xx = np.mgrid[:ctsim.N, :ctsim.N]
    r = np.hypot(xx - ctsim.CENTER, yy - ctsim.CENTER).astype(int)
    prof = np.array([img[r == k].mean() for k in range(r_max)])
    return prof


def ring_strength(img):
    """Deviation of the radial profile from its own smooth trend.
    A ring is a narrow bump, so it survives detrending.
    Returns (strength, radius_of_strongest_deviation)."""
    prof = radial_profile(img)
    trend = median_filter(prof, size=21, mode="nearest")
    resid = (prof - trend)[5:]            # skip the tiny noisy centre bins
    k = int(np.argmax(np.abs(resid)))
    return float(np.abs(resid[k])), k + 5


def make_gain(channel_offset, error):
    g = np.ones(n_det)
    g[det_center + channel_offset] *= (1.0 + error)
    return g


# ------------------------------------------------------------ Test 1
# Ring radius must equal the channel's distance from the rotation centre.
OFFSET = 40
img_ring = ctsim.simulate_scan(sino, 1e5, rng, make_gain(OFFSET, 0.05))
img_ok = ctsim.simulate_scan(sino, 1e5, rng)
s, r_found = ring_strength(img_ring)
print(f"Test 1: faulty channel offset = {OFFSET} px")
print(f"        ring found at radius = {r_found} px   (strength {s:.5f})")

# ------------------------------------------------------------ Test 2
# Severity sweep at two dose levels, averaged over repeats.
errors = [0.0, 0.005, 0.01, 0.02, 0.05]
doses = [1e4, 1e5]
repeats = 5
results = {}
print("\nTest 2: metric response vs fault severity")
print(f"{'I0':>8} {'gain err':>9} {'uniformity':>12} {'ring strength':>14}")
for I0 in doses:
    results[I0] = {"u": [], "r": []}
    for e in errors:
        us, rs = [], []
        for _ in range(repeats):
            im = ctsim.simulate_scan(sino, I0, rng, make_gain(OFFSET, e))
            us.append(uniformity(im))
            rs.append(ring_strength(im)[0])
        results[I0]["u"].append(np.mean(us))
        results[I0]["r"].append(np.mean(rs))
        print(f"{I0:>8.0e} {e:>9.3f} {np.mean(us):>12.5f} {np.mean(rs):>14.5f}")

# ------------------------------------------------------------ plots
fig, ax = plt.subplots(1, 4, figsize=(19, 4.5))
vmin, vmax = MU = (0.006, 0.014)
ax[0].imshow(img_ok, cmap="gray", vmin=vmin, vmax=vmax)
ax[0].set_title("No fault (I0=1e5)")
ax[1].imshow(img_ring, cmap="gray", vmin=vmin, vmax=vmax)
ax[1].set_title(f"5% gain error, channel +{OFFSET}")
for a in ax[:2]:
    a.axis("off")

ax[2].plot(radial_profile(img_ok), label="no fault")
ax[2].plot(radial_profile(img_ring), label="5% gain error")
ax[2].axvline(OFFSET, ls="--", c="gray", label=f"expected radius {OFFSET}")
ax[2].set_xlabel("Radius (px)")
ax[2].set_ylabel("Mean value")
ax[2].set_title("Radial profile")
ax[2].legend()

for I0 in doses:
    ax[3].plot(np.array(errors) * 100, results[I0]["r"], "o-",
               label=f"ring strength, I0={I0:.0e}")
    ax[3].plot(np.array(errors) * 100, results[I0]["u"], "s--",
               label=f"uniformity, I0={I0:.0e}")
ax[3].set_xlabel("Gain error (%)")
ax[3].set_ylabel("Metric value")
ax[3].set_title("Metric response vs severity")
ax[3].legend(fontsize=7)

plt.tight_layout()
plt.savefig("ring_artifact.png", dpi=150)
print("\nSaved ring_artifact.png")