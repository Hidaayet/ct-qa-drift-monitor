"""
ctsim.py
Minimal parallel-beam CT simulator with fault injection.
Reused by all experiment scripts. Later this moves to src/ctqa/simulation/.

Limitations (document in the project): parallel-beam, monoenergetic source,
no scatter, no electronic noise. Beam hardening is modelled with a simple
empirical nonlinearity, not a polychromatic spectrum.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from skimage.transform import radon, iradon

N = 256
MU_WATER = 0.010
ANGLES = np.linspace(0.0, 180.0, 360, endpoint=False)
CENTER = N // 2


def _disk(x0, y0, r):
    yy, xx = np.mgrid[:N, :N]
    return (xx - x0) ** 2 + (yy - y0) ** 2 <= r ** 2


def make_phantom(with_inserts=True):
    """Water cylinder (radius 100 px). Inserts are off-axis so the
    centre stays uniform."""
    p = np.zeros((N, N))
    p[_disk(CENTER, CENTER, 100)] = MU_WATER
    if with_inserts:
        p[_disk(CENTER + 55, CENTER, 12)] = MU_WATER * 1.5
        p[_disk(CENTER - 55, CENTER, 12)] = 0.0
        p[_disk(CENTER, CENTER + 55, 12)] = MU_WATER * 1.05
    return p


def forward(phantom):
    """Sinogram: shape (n_detector_channels, n_angles)."""
    return radon(phantom, theta=ANGLES)


def to_hu(img):
    """Attenuation -> CT number: water = 0 HU, air = -1000 HU."""
    return 1000.0 * (img - MU_WATER) / MU_WATER


def simulate_scan(sinogram, I0, rng, channel_gain=None,
                  blur_sigma=0.0, bh_k=0.0, hu_shift=0.0):
    """Noisy reconstruction from a sinogram, with optional faults.

    channel_gain : per-detector-channel gain (ring artifact).
    blur_sigma   : Gaussian blur along the detector axis, in channels
                   (focal-spot growth / resolution loss).
    bh_k         : beam-hardening strength. Measured p' = p - k*p^2, so long
                   paths are under-measured -> cupping.
    hu_shift     : calibration error, added to the image in HU.
    """
    sino = sinogram
    if blur_sigma > 0:
        sino = gaussian_filter1d(sino, blur_sigma, axis=0, mode="nearest")
    if bh_k > 0:
        sino = sino - bh_k * sino ** 2

    expected = I0 * np.exp(-sino)
    if channel_gain is not None:
        expected = expected * channel_gain[:, None]
    counts = np.maximum(rng.poisson(expected), 1)
    noisy_sino = -np.log(counts / I0)
    img = iradon(noisy_sino, theta=ANGLES, filter_name="ramp")

    if hu_shift != 0.0:
        img = img + hu_shift / 1000.0 * MU_WATER
    return img