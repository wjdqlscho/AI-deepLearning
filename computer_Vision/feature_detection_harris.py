# -*- coding: utf-8 -*-
"""Feature_Detection_Harris

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/10GtYK1f8llpVraa-vxNUbmQqjcJPAFCW
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("/content/rabbit4.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

block_size = 2
ksize = 3
k = 0.04
dst = cv2.cornerHarris(gray, block_size, ksize, k)

dst_dilate = cv2.dilate(dst, None)

threshold = 0.01 * dst_dilate.max()
img[dst > threshold] = [0, 0, 255]

plt.figure(figsize = (10, 10))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Harris corner detection")
plt.axis("off")
plt.show()

import cv2
import matplotlib.pyplot as plt

img_sift = cv2.imread("/content/rabbit4.jpg")
gray_sift = cv2.cvtColor(img_sift, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp_sift, des_sift = sift.detectAndCompute(gray_sift, None)

img_sift_kp = cv2.drawKeypoints(
    img_sift,
    kp_sift,
    None,
    flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(img_sift_kp, cv2.COLOR_BGR2RGB))
plt.title("SIFT keypoints")
plt.axis("off")
plt.show()

print("SIFT Key Points : ", len(kp_sift))
print("Descriptor size (dim) : ", des_sift.shape)