"""
01_noise_vs_dose.py
First validation experiment for the CT QA drift monitor.

Question: does our simulated CT scanner behave physically?
Expectation: image noise (SD in a uniform region) scales as 1/sqrt(dose),
i.e. slope -0.5 on a log-log plot of noise vs photon count.

Simplifications (state these in the project limitations section):
  - parallel-beam geometry (skimage radon/iradon), not fan/cone beam
  - monoenergetic X-rays (no beam hardening)
  - no scatter, no detector electronic noise
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon

rng = np.random.default_rng(seed=42)  # fixed seed = reproducible results

# ---------------------------------------------------------------- PIECE A
# Water cylinder phantom with a few inserts.
# Pixel values are linear attenuation in "per pixel" units.
N = 256
MU_WATER = 0.010
yy, xx = np.mgrid[:N, :N]
cx = cy = N // 2

def disk(x0, y0, r):
    return (xx - x0) ** 2 + (yy - y0) ** 2 <= r ** 2

phantom = np.zeros((N, N))
phantom[disk(cx, cy, 100)] = MU_WATER                    # water cylinder
phantom[disk(cx + 55, cy, 12)] = MU_WATER * 1.5          # dense insert (bone-like)
phantom[disk(cx - 55, cy, 12)] = MU_WATER * 0.0          # air insert
phantom[disk(cx, cy + 55, 12)] = MU_WATER * 1.05         # low-contrast insert

# Uniform region of interest (ROI) in the centre, away from the inserts
ROI = (slice(cy - 15, cy + 15), slice(cx - 15, cx + 15))

# ---------------------------------------------------------------- PIECE B
# Forward projection: the sinogram (line integrals of attenuation)
angles = np.linspace(0.0, 180.0, 360, endpoint=False)
sinogram = radon(phantom, theta=angles)

# ---------------------------------------------------------------- PIECE C
# Noise-free reconstruction as a sanity check
recon_clean = iradon(sinogram, theta=angles, filter_name="ramp")

# ---------------------------------------------------------------- PIECE D
# Noise lives in the PHOTON COUNTS, so: line integrals -> counts (Beer-Lambert)
# -> Poisson noise -> back to line integrals -> reconstruct.
def simulate_scan(I0):
    expected_counts = I0 * np.exp(-sinogram)
    noisy_counts = rng.poisson(expected_counts)
    noisy_counts = np.maximum(noisy_counts, 1)           # avoid log(0)
    noisy_sino = -np.log(noisy_counts / I0)
    return iradon(noisy_sino, theta=angles, filter_name="ramp")

# ---------------------------------------------------------------- PIECE E
# Sweep the dose (I0 = photons per detector ray, a proxy for mAs)
I0_values = np.array([1e3, 3e3, 1e4, 3e4, 1e5, 3e5])
n_repeats = 3
noise = []
for I0 in I0_values:
    sds = [simulate_scan(I0)[ROI].std() for _ in range(n_repeats)]
    noise.append(np.mean(sds))
    print(f"I0 = {I0:>9.0f}   noise (SD) = {noise[-1]:.6f}")
noise = np.array(noise)

# Fit a line in log-log space: slope should be close to -0.5
slope, intercept = np.polyfit(np.log10(I0_values), np.log10(noise), 1)
print(f"\nFitted log-log slope: {slope:.3f}   (theory: -0.5)")

# ---------------------------------------------------------------- PLOTS
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

axes[0].imshow(phantom, cmap="gray")
axes[0].set_title("Phantom (attenuation)")
axes[0].axis("off")

axes[1].imshow(simulate_scan(I0_values[0]), cmap="gray")
axes[1].set_title(f"Reconstruction, low dose (I0={I0_values[0]:.0e})")
axes[1].axis("off")

axes[2].loglog(I0_values, noise, "o-", label="simulated")
ref = noise[0] * (I0_values / I0_values[0]) ** -0.5
axes[2].loglog(I0_values, ref, "--", label="theory: slope -0.5")
axes[2].set_xlabel("Photons per ray (I0, dose proxy)")
axes[2].set_ylabel("Image noise (SD in uniform ROI)")
axes[2].set_title(f"Noise vs dose, fitted slope = {slope:.2f}")
axes[2].legend()
axes[2].grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig("noise_vs_dose.png", dpi=150)
print("Saved noise_vs_dose.png")
plt.show()