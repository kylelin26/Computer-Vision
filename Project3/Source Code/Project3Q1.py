import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# helpers
def read_gray_pil(path):
    """Read image with PIL, convert to 8-bit grayscale, return float64 array [0..255]."""
    img = Image.open(path).convert("L")   # force grayscale
    arr = np.array(img, dtype=np.float64)
    return arr

def save_gray_pil(arr, path):
    """Save a float/uint image array to disk via PIL (clipped to [0,255])."""
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)

def gaussian_kernel(M, sigma=None):
    """MxM Gaussian with sum normalized to 1. M must be odd."""
    assert M % 2 == 1, "M must be odd"
    if sigma is None:
        sigma = M / 4.0
    r = M // 2
    k = np.zeros((M, M), dtype=np.float64)
    s2 = 2.0 * sigma * sigma
    for i in range(-r, r+1):
        for j in range(-r, r+1):
            k[i+r, j+r] = np.exp(-(i*i + j*j) / s2)
    s = k.sum()
    if s != 0:
        k /= s
    return k

def mean_kernel(M):
    """MxM mean filter with sum normalized to 1."""
    assert M % 2 == 1, "M must be odd"
    val = 1.0 / (M * M)
    k = np.empty((M, M), dtype=np.float64)
    for i in range(M):
        for j in range(M):
            k[i, j] = val
    return k

def convolve_interior(img, kernel):
    """
    Convolve img with kernel ONLY on interior pixels:
    interior indices i,j in [r, N-1-r]. Border is left 0.
    """
    M = kernel.shape[0]
    r = M // 2
    H, W = img.shape
    out = np.zeros_like(img, dtype=np.float64)

    for i in range(r, H - r):
        for j in range(r, W - r):
            acc = 0.0
            # h(i,j) = sum_{k=0..M-1} sum_{l=0..M-1} g[k,l] * f(i-(k-r), j-(l-r))
            for k in range(M):
                for l in range(M):
                    ii = i - (k - r)
                    jj = j - (l - r)
                    acc += kernel[k, l] * img[ii, jj]
            out[i, j] = acc

    return np.clip(out, 0, 255)

# code for parameter
M = 9                 # odd: 3,5,7,9,...
SIGMA = M / 4.0       # per instructions
paths = [
    "pic1grey300.jpg",
    "pic2grey300.jpg",
]

# running the code
imgs = [read_gray_pil(p) for p in paths]

g_kernel = gaussian_kernel(M, SIGMA)
m_kernel = mean_kernel(M)

gauss_out = [convolve_interior(im, g_kernel) for im in imgs]
mean_out  = [convolve_interior(im, m_kernel) for im in imgs]


# display
fig, axes = plt.subplots(3, 2, figsize=(10, 12))

axes[0, 0].imshow(imgs[0], cmap="gray"); axes[0, 0].set_title("Original Image 1"); axes[0, 0].axis("off")
axes[0, 1].imshow(imgs[1], cmap="gray"); axes[0, 1].set_title("Original Image 2"); axes[0, 1].axis("off")

axes[1, 0].imshow(gauss_out[0], cmap="gray"); axes[1, 0].set_title(f"pic1 - Gaussian filter"); axes[1, 0].axis("off")
axes[1, 1].imshow(gauss_out[1], cmap="gray"); axes[1, 1].set_title(f"pic2 - Gaussian filter"); axes[1, 1].axis("off")

axes[2, 0].imshow(mean_out[0], cmap="gray"); axes[2, 0].set_title(f"pic1 - Mean filter"); axes[2, 0].axis("off")
axes[2, 1].imshow(mean_out[1], cmap="gray"); axes[2, 1].set_title(f"pic2 - Mean filter"); axes[2, 1].axis("off")

plt.tight_layout()
plt.show()
