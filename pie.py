
import cv2 as cv
import numpy as np
import scipy.sparse.linalg as splin
import time

src = cv.imread('boat.jpg')
bg = cv.imread('lake_bg.jpg')

# src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
# bg = cv.cvtColor(bg, cv.COLOR_BGR2GRAY)

h_1, w_1 = int(src.shape[0] * 0.25), int(src.shape[1] * 0.25)
h_2, w_2 = int(bg.shape[0] * 0.15), int(bg.shape[1] * 0.15)
src = cv.resize(src, (w_1, h_1))
bg = cv.resize(bg, (w_2, h_2))
bg_copy = bg.copy()
cv.imshow('Source', src)

cx = []
cy = []  

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        xy = "%d, %d" % (x, y)
        cx.append(x)
        cy.append(y)
        cv.circle(bg_copy, (x, y), 1, (0, 0, 255), thickness=-1)
        cv.putText(bg_copy, xy, (x, y), cv.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv.imshow("image", bg_copy)
        print(x,y)
 
cv.namedWindow("image")
cv.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv.imshow("image", bg_copy)
cv.waitKey(0)
print('Selected center coordinates: ', cx, cy)
upperL_x, upperL_y = int(cx[0] - w_1 / 2), int(cy[0] - h_1 / 2)
upperR_x, upperR_y = int(cx[0] + w_1 / 2), int(cy[0] - h_1 / 2)
bottomL_x, bottomL_y = int(cx[0] - w_1 / 2), int(cy[0] + h_1 / 2)
bottomR_x, bottomR_y = int(cx[0] + w_1 / 2), int(cy[0] + h_1 / 2)
src = src / 255
bg = bg / 255
b_src, g_src, r_src = cv.split(src)
b_bg, g_bg, r_bg = cv.split(bg)
b_fg, g_fg, r_fg = b_src.copy(), g_src.copy(), r_src.copy()

# Border
for i in range(0, w_1):
    b_fg[0, i] = b_bg[upperL_y, upperL_x + i]
    g_fg[0, i] = g_bg[upperL_y, upperL_x + i]
    r_fg[0, i] = r_bg[upperL_y, upperL_x + i]
    b_fg[h_1 - 1, i] = b_bg[bottomL_y, bottomL_x + i]
    g_fg[h_1 - 1, i] = g_bg[bottomL_y, bottomL_x + i]
    r_fg[h_1 - 1, i] = r_bg[bottomL_y, bottomL_x + i]

for i in range(1, h_1 - 1):
    b_fg[i, 0] = b_bg[upperL_y + i, upperL_x]
    g_fg[i, 0] = g_bg[upperL_y + i, upperL_x]
    r_fg[i, 0] = r_bg[upperL_y + i, upperL_x]
    b_fg[i, w_1 - 1] = b_bg[upperR_y + i, upperR_x]
    g_fg[i, w_1 - 1] = g_bg[upperR_y + i, upperR_x]
    r_fg[i, w_1 - 1] = r_bg[upperR_y + i, upperR_x]

wfact, hfact = w_1 - 2, h_1 - 2
borderlessSize = wfact * hfact
b_A = np.zeros([borderlessSize, borderlessSize], dtype=np.float32)
g_A = np.zeros([borderlessSize, borderlessSize], dtype=np.float32)
r_A = np.zeros([borderlessSize, borderlessSize], dtype=np.float32)
b_b = np.zeros([borderlessSize, 1], dtype=np.float32)
g_b = np.zeros([borderlessSize, 1], dtype=np.float32)
r_b = np.zeros([borderlessSize, 1], dtype=np.float32)

# b
for i in range(1, h_1 - 1):
    for j in range(1, w_1 - 1):
        b_b[(i - 1) * wfact + j - 1, 0] = b_src[i - 1, j] + b_src[i + 1, j] + b_src[i, j - 1] + b_src[i, j + 1] - 4 * b_src[i, j]
        g_b[(i - 1) * wfact + j - 1, 0] = g_src[i - 1, j] + g_src[i + 1, j] + g_src[i, j - 1] + g_src[i, j + 1] - 4 * g_src[i, j]
        r_b[(i - 1) * wfact + j - 1, 0] = r_src[i - 1, j] + r_src[i + 1, j] + r_src[i, j - 1] + r_src[i, j + 1] - 4 * r_src[i, j]
        if i == 1 and j == 1:
            b_b[(i - 1) * wfact + j - 1, 0] -= (b_fg[i - 1, j] + b_fg[i, j - 1])
            g_b[(i - 1) * wfact + j - 1, 0] -= (g_fg[i - 1, j] + g_fg[i, j - 1])
            r_b[(i - 1) * wfact + j - 1, 0] -= (r_fg[i - 1, j] + r_fg[i, j - 1])
        elif i == 1 and j == wfact:
            b_b[(i - 1) * wfact + j - 1, 0] -= (b_fg[i - 1, j] + b_fg[i, j + 1])
            g_b[(i - 1) * wfact + j - 1, 0] -= (g_fg[i - 1, j] + g_fg[i, j + 1])
            r_b[(i - 1) * wfact + j - 1, 0] -= (r_fg[i - 1, j] + r_fg[i, j + 1])
        elif i == hfact and j == 1:
            b_b[(i - 1) * wfact + j - 1, 0] -= (b_fg[i + 1, j] + b_fg[i, j - 1])
            g_b[(i - 1) * wfact + j - 1, 0] -= (g_fg[i + 1, j] + g_fg[i, j - 1])
            r_b[(i - 1) * wfact + j - 1, 0] -= (r_fg[i + 1, j] + r_fg[i, j - 1])
        elif i == hfact and j == wfact:
            b_b[(i - 1) * wfact + j - 1, 0] -= (b_fg[i + 1, j] + b_fg[i, j + 1])
            g_b[(i - 1) * wfact + j - 1, 0] -= (g_fg[i + 1, j] + g_fg[i, j + 1])
            r_b[(i - 1) * wfact + j - 1, 0] -= (r_fg[i + 1, j] + r_fg[i, j + 1])
        # Four corners
        # Four sides
        elif i == 1 and j > 1 and j < wfact:
            b_b[(i - 1) * wfact + j - 1, 0] -= b_fg[i - 1, j]
            g_b[(i - 1) * wfact + j - 1, 0] -= g_fg[i - 1, j]
            r_b[(i - 1) * wfact + j - 1, 0] -= r_fg[i - 1, j]
        elif i == hfact and j > 1 and j < wfact:
            b_b[(i - 1) * wfact + j - 1, 0] -= b_fg[i + 1, j]
            g_b[(i - 1) * wfact + j - 1, 0] -= g_fg[i + 1, j]
            r_b[(i - 1) * wfact + j - 1, 0] -= r_fg[i + 1, j]
        elif i > 1 and i < hfact and j == 1:
            b_b[(i - 1) * wfact + j - 1, 0] -= b_fg[i, j - 1]
            g_b[(i - 1) * wfact + j - 1, 0] -= g_fg[i, j - 1]
            r_b[(i - 1) * wfact + j - 1, 0] -= r_fg[i, j - 1]
        elif i > 1 and i < hfact and j == wfact:
            b_b[(i - 1) * wfact + j - 1, 0] -= b_fg[i, j + 1]
            g_b[(i - 1) * wfact + j - 1, 0] -= g_fg[i, j + 1]
            r_b[(i - 1) * wfact + j - 1, 0] -= r_fg[i, j + 1]

# A
for i in range(0, borderlessSize):
    b_A[i, i] = g_A[i, i] = r_A[i, i] = -4
    if i == 0:
        b_A[i, i + 1] = g_A[i, i + 1] = r_A[i, i + 1] = 1
        b_A[i + 1, i] = g_A[i + 1, i] = r_A[i + 1, i] = 1
        b_A[i, i + wfact] = g_A[i, i + wfact] = r_A[i, i + wfact] = 1
        b_A[i + wfact, i] = g_A[i + wfact, i] = r_A[i + wfact, i] = 1
    elif i == wfact - 1:
        b_A[i, i - 1] = g_A[i, i - 1] = r_A[i, i - 1] = 1
        b_A[i - 1, i] = g_A[i - 1, i] = r_A[i - 1, i] = 1
        b_A[i, i + wfact] = g_A[i, i + wfact] = r_A[i, i + wfact] = 1
        b_A[i + wfact, i] = g_A[i + wfact, i] = r_A[i + wfact, i] = 1
    elif i == (hfact - 1) * wfact:
        b_A[i, i + 1] = g_A[i, i + 1] = r_A[i, i + 1] = 1
        b_A[i + 1, i] = g_A[i + 1, i] = r_A[i + 1, i] = 1
        b_A[i, i - wfact] = g_A[i, i - wfact] = r_A[i, i - wfact] = 1
        b_A[i - wfact, i] = g_A[i - wfact, i] = r_A[i - wfact, i] = 1
    elif i == borderlessSize - 1:
        b_A[i, i - 1] = g_A[i, i - 1] = r_A[i, i - 1] = 1
        b_A[i - 1, i] = g_A[i - 1, i] = r_A[i - 1, i] = 1
        b_A[i, i - wfact] = g_A[i, i - wfact] = r_A[i, i - wfact] = 1
        b_A[i - wfact, i] = g_A[i - wfact, i] = r_A[i - wfact, i] = 1
    elif i > 0 and i < wfact - 1:
        b_A[i, i - 1] = g_A[i, i - 1] = r_A[i, i - 1] = 1
        b_A[i - 1, i] = g_A[i - 1, i] = r_A[i - 1, i] = 1
        b_A[i, i + 1] = g_A[i, i + 1] = r_A[i, i + 1] = 1
        b_A[i + 1, i] = g_A[i + 1, i] = r_A[i + 1, i] = 1
        b_A[i, i + wfact] = g_A[i, i + wfact] = r_A[i, i + wfact] = 1
        b_A[i + wfact, i] = g_A[i + wfact, i] = r_A[i + wfact, i] = 1
    elif i > (hfact - 1) * wfact and i < borderlessSize - 1:
        b_A[i, i - 1] = g_A[i, i - 1] = r_A[i, i - 1] = 1
        b_A[i - 1, i] = g_A[i - 1, i] = r_A[i - 1, i] = 1
        b_A[i, i + 1] = g_A[i, i + 1] = r_A[i, i + 1] = 1
        b_A[i + 1, i] = g_A[i + 1, i] = r_A[i + 1, i] = 1
        b_A[i, i - wfact] = g_A[i, i - wfact] = r_A[i, i - wfact] = 1
        b_A[i - wfact, i] = g_A[i - wfact, i] = r_A[i - wfact, i] = 1
    elif i > 0 and i % wfact == 0 and i < (hfact - 1) * wfact:
        b_A[i, i + 1] = g_A[i, i + 1] = r_A[i, i + 1] = 1
        b_A[i + 1, i] = g_A[i + 1, i] = r_A[i + 1, i] = 1
        b_A[i, i - wfact] = g_A[i, i - wfact] = r_A[i, i - wfact] = 1
        b_A[i - wfact, i] = g_A[i - wfact, i] = r_A[i - wfact, i] = 1
        b_A[i, i + wfact] = g_A[i, i + wfact] = r_A[i, i + wfact] = 1
        b_A[i + wfact, i] = g_A[i + wfact, i] = r_A[i + wfact, i] = 1
    elif i > wfact - 1 and i % wfact == wfact - 1 and i < borderlessSize - 1:
        b_A[i, i - 1] = g_A[i, i - 1] = r_A[i, i - 1] = 1
        b_A[i - 1, i] = g_A[i - 1, i] = r_A[i - 1, i] = 1
        b_A[i, i - wfact] = g_A[i, i - wfact] = r_A[i, i - wfact] = 1
        b_A[i - wfact, i] = g_A[i - wfact, i] = r_A[i - wfact, i] = 1
        b_A[i, i + wfact] = g_A[i, i + wfact] = r_A[i, i + wfact] = 1
        b_A[i + wfact, i] = g_A[i + wfact, i] = r_A[i + wfact, i] = 1
    else:
        b_A[i, i - 1] = g_A[i, i - 1] = r_A[i, i - 1] = 1
        b_A[i - 1, i] = g_A[i - 1, i] = r_A[i - 1, i] = 1
        b_A[i, i + 1] = g_A[i, i + 1] = r_A[i, i + 1] = 1
        b_A[i + 1, i] = g_A[i + 1, i] = r_A[i + 1, i] = 1
        b_A[i, i - wfact] = g_A[i, i - wfact] = r_A[i, i - wfact] = 1
        b_A[i - wfact, i] = g_A[i - wfact, i] = r_A[i - wfact, i] = 1
        b_A[i, i + wfact] = g_A[i, i + wfact] = r_A[i, i + wfact] = 1
        b_A[i + wfact, i] = g_A[i + wfact, i] = r_A[i + wfact, i] = 1

b_x = splin.spsolve(b_A, b_b)
g_x = splin.spsolve(g_A, g_b)
r_x = splin.spsolve(r_A, r_b)

for i in range(1, h_1 - 1):
    for j in range(1, w_1 - 1):
        b_fg[i, j] = b_x[(i - 1) * (w_1 - 2) + j - 1]
        g_fg[i, j] = g_x[(i - 1) * (w_1 - 2) + j - 1]
        r_fg[i, j] = r_x[(i - 1) * (w_1 - 2) + j - 1]

fgMerged = cv.merge([b_fg, g_fg, r_fg])
bg[upperL_y:bottomL_y, upperL_x:upperR_x] = fgMerged
'''
if i < upperL_y + h_1 - 1 and j < upperL_x + w_1 - 1:
     if bg[i + 1, j] + bg[i, j + 1] - 2 * bg[i, j] < fg[i - upperL_y + 1, j - upperL_x] + fg[i - upperL_y, j - upperL_x + 1] - 2 * fg[i - upperL_y, j - upperL_x]:
        bg[i, j] = bg[i, j]
    else: bg[i, j] = fg[i - upperL_y, j - upperL_x]
else: bg[i, j] = fg[i - upperL_y, j - upperL_x]
'''

cv.namedWindow('Result', cv.WINDOW_AUTOSIZE)
cv.imshow('Result', bg)
cv.waitKey(0)


            


