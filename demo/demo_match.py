"""
https://blog.csdn.net/Eddy_zheng/article/details/78916009
"""
import cv2
import matplotlib.pyplot as plt
import os

city = 'beijing'
img = cv2.imread('data/match/beijing.png', cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create()

for i in range(1, 5):  # 1-4 NEWS
    img_path = f'data/stations/{city}/{i}.png'
    if not os.path.exists(img_path):
        continue

    # 匹配的 基准 img
    base_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h, w = base_img.shape
    base_img = cv2.resize(base_img, dsize=(400, int(400 * h / w)))

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img, None)
    kp2, des2 = sift.detectAndCompute(base_img, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = [[m] for m, n in matches if m.distance < 0.7 * n.distance]
    print(len(good))

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img, kp1, base_img, kp2, good, None, flags=2)
    print(img3.dtype)

    plt.imshow(img3)
    plt.show()
