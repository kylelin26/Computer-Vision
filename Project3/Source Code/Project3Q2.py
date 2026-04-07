import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 1D gaussian (sum = 1)
def gaussian_filter_1d(M: int, sigma: float) -> np.ndarray:
    assert M % 2 == 1, "M must be odd"
    half = M // 2
    k = np.arange(M) - half
    g = np.exp(-(k**2) / (2.0 * sigma * sigma))
    g /= g.sum()
    return g.astype(np.float32)

#  row then column filtering (interior only, borders left 0) 
def filter_rows(image: np.ndarray, g: np.ndarray) -> np.ndarray:
    M = g.size
    H, W = image.shape
    pad = M // 2
    h1 = np.zeros_like(image, dtype=np.float32)
    for i in range(H):
        for j in range(pad, W - pad):
            s = 0.0
            for k in range(M):
                s += g[k] * image[i, j - (k - pad)]
            h1[i, j] = s
    return h1

def filter_columns(h1: np.ndarray, g: np.ndarray) -> np.ndarray:
    M = g.size
    H, W = h1.shape
    pad = M // 2
    h2 = np.zeros_like(h1, dtype=np.float32)
    for j in range(W):
        for i in range(pad, H - pad):
            s = 0.0
            for k in range(M):
                s += g[k] * h1[i - (k - pad), j]
            h2[i, j] = s
    return h2

# gradient magnitude and contrast stretch 
def gradient_mag_0_255(img: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(img, dtype=np.float32)
    gy = np.zeros_like(img, dtype=np.float32)
    gx[:-1, :] = img[1:, :] - img[:-1, :]
    gy[:, :-1] = img[:, 1:] - img[:, :-1]
    mag = np.sqrt(gx * gx + gy * gy)

    # zero borders explicitly
    mag[0, :] = mag[-1, :] = 0
    mag[:, 0] = mag[:, -1] = 0

    # stretch to 0 - 255 using the 99th percentile to avoid outliers
    p99 = np.percentile(mag, 99.0)
    scale = 255.0 / (p99 + 1e-6)
    mag = np.clip(mag * scale, 0, 255)
    return mag.astype(np.float32)

def threshold_edges(mag_0255: np.ndarray, T: float) -> np.ndarray:
    out = np.zeros_like(mag_0255, dtype=np.uint8)
    out[mag_0255 > T] = 255
    return out

def run_one(img: np.ndarray, sigma: float, thresholds):
    M = 9  # assignment: 9x9 gaussian
    g = gaussian_filter_1d(M, sigma)
    h1 = filter_rows(img, g)
    h2 = filter_columns(h1, g)
    mag = gradient_mag_0_255(h2)  # normalized to 0 - 255

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for ax, T in zip(axes.ravel(), thresholds):
        edges = threshold_edges(mag, T)
        ax.imshow(edges, cmap="gray")
        ax.set_title(f"σ={sigma}, T={T}")
        ax.axis("off")
    fig.tight_layout()
    plt.show()

img1 = np.array(Image.open("pic1grey300.jpg").convert("L"), dtype=np.float32)
img2 = np.array(Image.open("pic2grey300.jpg").convert("L"), dtype=np.float32)

sigmas = [1.0, 2.0, 3.0]
thresholds = [20, 30, 40, 50]

for s in sigmas:
    run_one(img1, s, thresholds)

for s in sigmas:
    run_one(img2, s, thresholds)
