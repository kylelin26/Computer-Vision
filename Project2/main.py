
"""
ESE 358 Project 2
Original Matlab project by: M. Subbarao, ECE, SBU
Python Template : Revised by TA C. Orlassino (8/31/2022)

TA Yucheng Xing updates on 9/21/2023:
    This is an updated version of the project 2 template. 
    Please check lines 69-73 and lines 98-105. 
    There are some extra clues that may help you avoid minor errors 
    on the cube's direction of motion. 

Stable version: python 3.8

Don't touch the import statements. The template uses numpy and cv2.

Installing necessary packages:
* Many IDEs will prompt you to install automatically on a failed import
* If not, run from console in this directory: "pip install opencv-python"
* Try "pip3" or "py -m pip" if "pip" doesn't work for you 
* ^This depends on your environment variables set when you installed python

More info on necessary packages:
cv2 package: https://pypi.org/project/opencv-python/
numpy package: https://numpy.org/install/
"""

import sys
import numpy as np
import cv2

'''
function for rotation and translation
'''
def Map2Da(K, R, T, Vi):
    T_transpose = np.transpose(np.atleast_2d(T)) #numpy needs to treat 1D as 2D to transpose
    V_transpose = np.transpose(np.atleast_2d(np.append(Vi,[1])))
    RandTappended = np.append(R, T_transpose, axis=1)
    P = K @ RandTappended @ V_transpose #@ is the matrix mult operator for numpy arrays
    P = np.asarray(P).flatten() #just to make it into a flat array

    w1 = P[2]
    v= [None]*2 #makes an empty array of size 2

    #map Vi = (X, Y, Z) to v = (x, y)
    v[0]= P[0] / w1  #v[0] is the x-value for the 2D point v
    v[1] = P[1] / w1 # v[1] is the y-value

    return v

'''
function for mapping image coordinates in mm to
row and column index of the image, with pixel size p mm and
image center at [r0,c0]
'''
def MapIndex(u, c0, r0, p):
    v = [None]*2
    v[0] = round(r0 - u[1] / p)
    v[1] = round(c0 + u[0] / p)
    return v

'''
Manual line drawing (replaces cv2.line)
'''
def drawLine(A, vertex1, vertex2, color=(255, 255, 255), thickness=1):
    r1, c1 = int(vertex1[0]), int(vertex1[1])
    r2, c2 = int(vertex2[0]), int(vertex2[1])
    dr, dc = (r2 - r1), (c2 - c1)
    d = int(np.hypot(dr, dc))
    if d == 0:
        if 0 <= r1 < A.shape[0] and 0 <= c1 < A.shape[1]:
            A[r1, c1] = color
        return A
    ur, uc = dr / max(d, 1e-9), dc / max(d, 1e-9)
    step = 0.5
    for k in np.arange(0, d + step, step):
        rr = int(round(r1 + k * ur))
        cc = int(round(c1 + k * uc))
        if 0 <= rr < A.shape[0] and 0 <= cc < A.shape[1]:
            A[rr, cc] = color
            for t in range(1, int(max(1, thickness))):
                for (offr, offc) in [(-t,0),(t,0),(0,-t),(0,t)]:
                    rrt, cct = rr + offr, cc + offc
                    if 0 <= rrt < A.shape[0] and 0 <= cct < A.shape[1]:
                        A[rrt, cct] = color
    return A

def main():
    length = 10 #length of an edge in mm
    V1 = np.array([0, 0, 0])
    V2 = np.array([0, length, 0])
    V3 = np.array([length, length, 0])
    V4 = np.array([length, 0, 0])
    V5 = np.array([length, 0, length])
    V6 = np.array([0, length, length])
    V7 = np.array([0, 0, length])
    V8 = np.array([length, length, length])

    #axis of rotation
    u81 = (V8 - V1) / np.linalg.norm(V8 - V1)
    ux, uy, uz = u81
    N = np.array([[0, -uz, uy],
                  [uz, 0, -ux],
                  [-uy, ux, 0]], dtype=np.float64)

    #initialized values
    T0 = np.array([-20, -25, 500])
    f = 40
    velocity = np.array([2, 9, 7])
    acc = np.array([0.0, -0.80, 0])
    theta0 = 0
    w0 = 20
    p = 0.01
    Rows = 600
    Cols = 600
    r0 = np.round(Rows / 2)
    c0 = np.round(Cols / 2)
    time_range = np.arange(0.0, 24.2, 0.2)

    #intrinsic K
    K = np.array([[f, 0, 0],
              [0, f, 0],
              [0, 0, 1]], dtype=np.float64)

    #texture face setup
    h = np.linalg.norm(V2 - V1)
    w = np.linalg.norm(V4 - V1)
    u21 = (V2 - V1) / h
    u41 = (V4 - V1) / w

    tmap = cv2.imread('einstein50x50v.jpg')
    if tmap is None:
        print("texture not found")
        sys.exit(1)

    r, c, colors = tmap.shape
    X = np.zeros((r, c), dtype=np.float64)
    Y = np.zeros((r, c), dtype=np.float64)
    Z = np.zeros((r, c), dtype=np.float64)
    for i in range(0, r):
        for j in range(0, c):
            p1 = V1 + (i) * u21 * (h / r) + (j) * u41 * (w / c)
            X[i, j] = p1[0]
            Y[i, j] = p1[1]
            Z[i, j] = p1[2]

    for t in time_range:
        theta = theta0 + w0 * t
        T = T0 + velocity * t + 0.5 * acc * t * t
        theta_rad = np.deg2rad(theta)
        R = np.eye(3) + np.sin(theta_rad) * N + (1 - np.cos(theta_rad)) * (N @ N)

        v = Map2Da(K, R, T, V1)
        v1 = MapIndex(v, c0, r0, p)
        v2 = MapIndex(Map2Da(K, R, T, V2), c0, r0, p)
        v3 = MapIndex(Map2Da(K, R, T, V3), c0, r0, p)
        v4 = MapIndex(Map2Da(K, R, T, V4), c0, r0, p)
        v5 = MapIndex(Map2Da(K, R, T, V5), c0, r0, p)
        v6 = MapIndex(Map2Da(K, R, T, V6), c0, r0, p)
        v7 = MapIndex(Map2Da(K, R, T, V7), c0, r0, p)
        v8 = MapIndex(Map2Da(K, R, T, V8), c0, r0, p)

        try:
            bg = cv2.imread('background.jpg')         #comment lines 164 - to 167 to get black background
            if bg is not None:
                A = cv2.resize(bg, (Cols, Rows))
            else:
                A = np.zeros((Rows, Cols, 3), dtype=np.uint8)
        except Exception:
            A = np.zeros((Rows, Cols, 3), dtype=np.uint8)


        color = (0, 0, 255)
        thickness = 2
        A = drawLine(A, v1, v2, color, thickness)
        A = drawLine(A, v2, v3, color, thickness)
        A = drawLine(A, v3, v4, color, thickness)
        A = drawLine(A, v4, v1, color, thickness)
        A = drawLine(A, v7, v6, color, thickness)
        A = drawLine(A, v6, v8, color, thickness)
        A = drawLine(A, v8, v5, color, thickness)
        A = drawLine(A, v5, v7, color, thickness)
        A = drawLine(A, v1, v7, color, thickness)
        A = drawLine(A, v2, v6, color, thickness)
        A = drawLine(A, v3, v8, color, thickness)
        A = drawLine(A, v4, v5, color, thickness)

        for i in range(r):
            for j in range(c):
                p1 = [X[i, j], Y[i, j], Z[i, j]]
                v_tmp = Map2Da(K, R, T, p1)
                ir, jr = MapIndex(v_tmp, c0, r0, p)
                if ((ir >= 0) and (jr >= 0) and (ir < Rows) and (jr < Cols)):
                    tmapval = tmap[i, j, 2]
                    A[ir ,jr] = [tmapval, tmapval, tmapval]

        cv2.imshow("Display Window", A)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()

