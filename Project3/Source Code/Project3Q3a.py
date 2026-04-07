import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# helpers
def gaussian_kernel_1d(length: int, sigma: float) -> np.ndarray:
    assert length % 2 == 1, "Kernel length must be odd"
    half = length // 2
    k = np.arange(length) - half
    g = np.exp(-(k**2) / (2.0 * sigma * sigma))
    g /= g.sum()
    return g.astype(np.float32)

def separable_filter(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    H, W = img.shape
    M = k.size
    pad = M // 2
    tmp = np.zeros_like(img, dtype=np.float32)
    out = np.zeros_like(img, dtype=np.float32)
    # rows
    for i in range(H):
        for j in range(pad, W - pad):
            tmp[i, j] = float(np.dot(k, img[i, j - pad:j + pad + 1]))
    # cols
    for j in range(W):
        for i in range(pad, H - pad):
            out[i, j] = float(np.dot(k, tmp[i - pad:i + pad + 1, j]))
    return out

# gradients
def gradients_forward(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Ix = np.zeros_like(img, dtype=np.float32)
    Iy = np.zeros_like(img, dtype=np.float32)
    Ix[:-1, :] = img[1:, :] - img[:-1, :]
    Iy[:, :-1] = img[:, 1:] - img[:, :-1]
    Ix[0, :] = Ix[-1, :] = 0
    Iy[:, 0] = Iy[:, -1] = 0
    Ix /= 10.0
    Iy /= 10.0
    return Ix, Iy

def harris_R(Ix: np.ndarray, Iy: np.ndarray) -> np.ndarray:
    A = Ix * Ix
    B = Iy * Iy
    C = Ix * Iy
    k11 = gaussian_kernel_1d(11, 5.5)
    A_ = separable_filter(A, k11)
    B_ = separable_filter(B, k11)
    C_ = separable_filter(C, k11)
    detM = A_ * B_ - C_ * C_
    traceM = A_ + B_
    R = detM - 0.04 * (traceM * traceM)
    R[0, :] = R[-1, :] = 0
    R[:, 0] = R[:, -1] = 0
    return R

# fixed 3x3 nms
def nms_3x3(R: np.ndarray) -> np.ndarray:
    H, W = R.shape
    Rp = np.pad(R, 1, mode="constant", constant_values=-np.inf)
    neighbor_max = np.full((H, W), -np.inf, dtype=np.float32)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:  # skip center
                continue
            nb = Rp[1+di:1+di+H, 1+dj:1+dj+W]
            neighbor_max = np.maximum(neighbor_max, nb)
    return R > neighbor_max

# main, it returns 3 values
def harris_corners(img_uint8: np.ndarray, thresh_frac: float = 0.08):
    img = img_uint8.astype(np.float32)
    k9 = gaussian_kernel_1d(9, 2.0)         # 9x9, sigma=2
    sm = separable_filter(img, k9)
    Ix, Iy = gradients_forward(sm)
    R = harris_R(Ix, Iy)
    T = float(R.max()) * thresh_frac if R.max() > 0 else 0.0
    peaks = (R > T) & nms_3x3(R)

    out = img_uint8.copy()
    ys, xs = np.where(peaks)
    for y, x in zip(ys, xs):
        out[max(0, y-1):min(out.shape[0], y+2),
            max(0, x-1):min(out.shape[1], x+2)] = 255
    print(f"Max R={R.max():.3f} | T={T:.3f} | corners={len(xs)}")
    return out, peaks, R  # <- THREE values


img1 = np.array(Image.open("pic1grey300.jpg").convert("L"))
img2 = np.array(Image.open("pic2grey300.jpg").convert("L"))

vis1, peaks1, R1 = harris_corners(img1, thresh_frac=0.08)
vis2, peaks2, R2 = harris_corners(img2, thresh_frac=0.08)

fig, axs = plt.subplots(1,2,figsize=(10,5))
axs[0].imshow(vis1, cmap="gray"); axs[0].set_title("Image 1"); axs[0].axis("off")
axs[1].imshow(vis2, cmap="gray"); axs[1].set_title("Image 2"); axs[1].axis("off")
plt.tight_layout(); plt.show()
