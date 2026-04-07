import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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

# gradients (forward diff; divide by 10 per spec)
def gradients_forward(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    H, W = img.shape
    Ix = np.zeros_like(img, dtype=np.float32)
    Iy = np.zeros_like(img, dtype=np.float32)
    Ix[:-1, :] = img[1:, :] - img[:-1, :]
    Iy[:, :-1] = img[:, 1:] - img[:, :-1]
    Ix[0, :] = Ix[-1, :] = 0
    Iy[:, 0] = Iy[:, -1] = 0
    Ix /= 10.0
    Iy /= 10.0
    return Ix, Iy

# harris response and NMS 
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

def nms_3x3(R: np.ndarray) -> np.ndarray:
    H, W = R.shape
    Rp = np.pad(R, 1, mode="constant", constant_values=-np.inf)
    neighbor_max = np.full((H, W), -np.inf, dtype=np.float32)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            nb = Rp[1+di:1+di+H, 1+dj:1+dj+W]
            neighbor_max = np.maximum(neighbor_max, nb)
    return R > neighbor_max

def harris_corners_coords(img_uint8: np.ndarray, thresh_frac: float = 0.08):
    img = img_uint8.astype(np.float32)
    # initial smoothing 9x9 σ=2
    k9 = gaussian_kernel_1d(9, 2.0)
    sm = separable_filter(img, k9)
    Ix, Iy = gradients_forward(sm)
    R = harris_R(Ix, Iy)
    T = float(R.max()) * thresh_frac if R.max() > 0 else 0.0
    peaks = (R > T) & nms_3x3(R)
    ys, xs = np.where(peaks)
    print(f"Harris: Max R={R.max():.3f} | T={T:.3f} | corners={len(xs)}")
    return (ys, xs), R, Ix, Iy

# descriptor over 9x9 window
def descriptor_9x9_at(ys, xs, Ix, Iy):
    """
    for each (y,x) corner, compute 8-bin histogram of gradient directions within 9x9
    window centered at (y,x). Bins at 0,45, ... ,315 deg; voting by magnitude.
    then rotate so max bin lands at 180 deg (bin index 4).
    returns: list of (y, x, hist_rotated(np.ndarray length 8))
    """
    H, W = Ix.shape
    results = []
    r = 4  # radius for 9x9
    # precompute angle and magnitude images
    mag = np.sqrt(Ix*Ix + Iy*Iy)
    ang = (np.degrees(np.arctan2(Iy, Ix)) + 360.0) % 360.0  # [0,360)
    for y, x in zip(ys, xs):
        # skip corners too close to border (need full 9x9)
        if y - r < 0 or y + r >= H or x - r < 0 or x + r >= W:
            continue
        h = np.zeros(8, dtype=np.float32)
        # accumulate votes
        for yy in range(y - r, y + r + 1):
            for xx in range(x - r, x + r + 1):
                a = ang[yy, xx]
                m = mag[yy, xx]
                bin_idx = int(np.round(a / 45.0)) % 8
                h[bin_idx] += m
        # rotate so max moves to 180 degrees -> bin 4
        m_idx = int(np.argmax(h))
        shift = (4 - m_idx) % 8
        h_rot = np.roll(h, shift)
        results.append((y, x, h_rot))
    return results

def print_descriptors(results):
    for (y, x, h) in results:
        # format to one decimal place like the example
        vals = ", ".join(f"{v:.1f}" for v in h.tolist())
        print(f"pixel at (i,j)=({y},{x}) has histogram {vals}")

# load image(s), detect corners, compute and print descriptors 
if __name__ == "__main__":
    img1 = np.array(Image.open("pic1grey300.jpg").convert("L"))
    img2 = np.array(Image.open("pic2grey300.jpg").convert("L"))

    # get corners 
    (ys1, xs1), R1, Ix1, Iy1 = harris_corners_coords(img1, thresh_frac=0.06)
    (ys2, xs2), R2, Ix2, Iy2 = harris_corners_coords(img2, thresh_frac=0.06)

    # compute descriptors
    desc1 = descriptor_9x9_at(ys1, xs1, Ix1, Iy1)
    desc2 = descriptor_9x9_at(ys2, xs2, Ix2, Iy2)


    print("\n Image 1 descriptors ")
    print_descriptors(desc1)

    print("\n Image 2 descriptors ")
    print_descriptors(desc2)

