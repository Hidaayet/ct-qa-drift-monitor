"""
ctmetrics.py
QA metrics computed on a reconstructed water-cylinder phantom image
(attenuation units in, results in HU / pixels out).
Later this moves to src/ctqa/metrics/.

Changes vs 03_fault_library.py:
  - edge_width() measures the background from the image instead of
    assuming it is zero (removes the HU-drift cross-talk).
  - ring_norm = ring_hu / noise_hu, so the ring metric is not fooled by
    a noisier image (tube output decline).
"""
import numpy as np
from scipy.ndimage import median_filter
import ctsim

C = ctsim.CENTER
METRIC_NAMES = ["mean_hu", "noise_hu", "unif_hu", "ring_hu", "ring_norm", "edge_px"]


def _roi(img, x, y, half=12):
    return img[y - half:y + half, x - half:x + half]


def radial_profile(img, r_max=130):
    yy, xx = np.mgrid[:ctsim.N, :ctsim.N]
    r = np.hypot(xx - C, yy - C).astype(int)
    return np.array([img[r == k].mean() for k in range(r_max)])


def edge_width(prof):
    """10-90% width of the phantom edge (true edge at r=100).
    Plateau = water region, background = outside the phantom."""
    plateau = prof[80:92].mean()
    background = prof[118:128].mean()
    hi = background + 0.9 * (plateau - background)
    lo = background + 0.1 * (plateau - background)
    r = np.arange(len(prof))
    seg = slice(92, 125)
    p, rr = prof[seg], r[seg]

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
    ring = float(np.abs((prof_hu - trend)[5:]).max())
    noise = float(centre.std())
    return {
        "mean_hu": float(centre.mean()),
        "noise_hu": noise,
        "unif_hu": float(periph - centre.mean()),
        "ring_hu": ring,
        "ring_norm": ring / noise,
        "edge_px": float(edge_width(radial_profile(img_mu))),
    }