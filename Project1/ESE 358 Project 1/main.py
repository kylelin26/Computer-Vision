# main.py
# project 1: Accept any input image
# explicit pixel operations: channel split, grayscale, histogram, binary threshold, edges, pyramid

import os, sys, math
from typing import Tuple, List
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# helpers 
def ensure_dir(d: str):
    if not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def load_rgb_512(path: str) -> np.ndarray:
    """Load image as RGB and force 512x512."""
    img = Image.open(path).convert("RGB")
    img = img.resize((512, 512))
    return np.array(img, dtype=np.uint8)

def save_and_show_gray(arr: np.ndarray, title: str, outpath: str):
    plt.figure()
    plt.imshow(arr, cmap="gray", vmin=0, vmax=255)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    Image.fromarray(arr).save(outpath, format="JPEG")
    plt.close()

def save_and_show_rgb(arr: np.ndarray, title: str, outpath: str):
    plt.figure()
    plt.imshow(arr)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    Image.fromarray(arr).save(outpath, format="JPEG")
    plt.close()


def histogram_256_by_scan(img_u8: np.ndarray) -> List[int]:
    h, w = img_u8.shape
    hist = [0]*256
    for m in range(h):
        for n in range(w):
            hist[int(img_u8[m, n])] += 1
    return hist

def show_hist(hist: List[int], title: str, outpath: str):
    plt.figure()
    plt.bar(range(256), hist)
    plt.title(title)
    plt.xlabel("Brightness (0..255)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath.replace(".png", ".jpg"), format="jpeg")
    plt.show()
    plt.close()

# -- Required Parts --
def split_channels_gray(A_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return R, G, B as 2D grayscale arrays (uint8)."""
    R = A_rgb[:, :, 0].astype(np.uint8)
    G = A_rgb[:, :, 1].astype(np.uint8)
    B = A_rgb[:, :, 2].astype(np.uint8)
    return R, G, B

def colorize_single_channel(R: np.ndarray, G: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create RGB images where only one channel is present (others zero).
    Returns (R_rgb, G_rgb, B_rgb) each as HxWx3 uint8.
    """
    h, w = R.shape
    zeros = np.zeros((h, w), dtype=np.uint8)

    R_rgb = np.stack([R, zeros, zeros], axis=2)
    G_rgb = np.stack([zeros, G, zeros], axis=2)
    B_rgb = np.stack([zeros, zeros, B], axis=2)
    return R_rgb, G_rgb, B_rgb

def grayscale_average(R: np.ndarray, G: np.ndarray, B: np.ndarray) -> np.ndarray:
    h, w = R.shape
    AG = np.zeros((h, w), dtype=np.uint8)
    for m in range(h):
        for n in range(w):
            AG[m, n] = (int(R[m, n]) + int(G[m, n]) + int(B[m, n])) // 3
    return AG

def threshold_binary(AG: np.ndarray, TB: int) -> np.ndarray:
    h, w = AG.shape
    AB = np.zeros((h, w), dtype=np.uint8)
    for m in range(h):
        for n in range(w):
            AB[m, n] = 0 if int(AG[m, n]) < TB else 255
    return AB

def simple_edges(AG: np.ndarray, TE: int) -> np.ndarray:
    h, w = AG.shape
    AE = np.zeros((h, w), dtype=np.uint8)
    for m in range(h):
        for n in range(w):
            gx = (int(AG[m, n+1]) - int(AG[m, n])) if n < w-1 else 0
            gy = (int(AG[m+1, n]) - int(AG[m, n])) if m < h-1 else 0
            gm = math.sqrt(gx*gx + gy*gy)
            AE[m, n] = 255 if gm > TE else 0
    return AE

def downsample_2x2_avg(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    new_h, new_w = h // 2, w // 2
    out = np.zeros((new_h, new_w), dtype=np.uint8)
    for m in range(new_h):
        for n in range(new_w):
            r0, c0 = 2*m, 2*n
            s = (int(gray[r0, c0]) +
                 int(gray[r0, c0+1]) +
                 int(gray[r0+1, c0]) +
                 int(gray[r0+1, c0+1]))
            out[m, n] = s // 4
    return out

# -- Main --
def main():
    outputs = "outputs"
    ensure_dir(outputs)

    # Handle input
    if len(sys.argv) >= 2:
        img_path = sys.argv[1]
    else:
        img_path = input("Enter path to image file: ").strip()
    if not os.path.isfile(img_path):
        print(f"File not found: {img_path}")
        return

    # Load and display
    A = load_rgb_512(img_path)
    save_and_show_rgb(A, "Original Image (512x512)", os.path.join(outputs, "A.jpg"))

    #  Channel Split 
    # 2D grayscale planes for math/ops:
    R, G, B = split_channels_gray(A)

    # Colorized views (only that channel lit):
    R_rgb, G_rgb, B_rgb = colorize_single_channel(R, G, B)
    save_and_show_rgb(R_rgb, "Red Channel (colorized)",   os.path.join(outputs, "RC_color.jpg"))
    save_and_show_rgb(G_rgb, "Green Channel (colorized)", os.path.join(outputs, "GC_color.jpg"))
    save_and_show_rgb(B_rgb, "Blue Channel (colorized)",  os.path.join(outputs, "BC_color.jpg"))


    #  Grayscale (average) 
    AG = grayscale_average(R, G, B)
    save_and_show_gray(AG, "Grayscale AG", os.path.join(outputs, "AG.jpg"))

    #  Histograms 
    for arr, name in [(R,"R"),(G,"G"),(B,"B"),(AG,"AG")]:
        hist = histogram_256_by_scan(arr)
        show_hist(hist, f"Histogram of {name}", os.path.join(outputs, f"hist_{name}.png"))

    #  Binarization 
    try:
        TB = int(input("Enter TB (default 100): ").strip())
    except:
        TB = 100
    AB = threshold_binary(AG, TB)
    save_and_show_gray(AB, f"Binary AB (TB={TB})", os.path.join(outputs, f"AB_TB{TB}.jpg"))

    #  Edges 
    try:
        TE = int(input("Enter TE (default 15): ").strip())
    except:
        TE = 15
    AE = simple_edges(AG, TE)
    save_and_show_gray(AE, f"Edges AE (TE={TE})", os.path.join(outputs, f"AE_TE{TE}.jpg"))

    #  Pyramid 
    AG2 = downsample_2x2_avg(AG); save_and_show_gray(AG2, "AG2 (1/2 size)",  os.path.join(outputs, "AG2.jpg"))
    AG4 = downsample_2x2_avg(AG2); save_and_show_gray(AG4, "AG4 (1/4 size)", os.path.join(outputs, "AG4.jpg"))
    AG8 = downsample_2x2_avg(AG4); save_and_show_gray(AG8, "AG8 (1/8 size)", os.path.join(outputs, "AG8.jpg"))

    print("\nAll outputs saved in:", os.path.abspath(outputs))

if __name__ == "__main__":
    main()
