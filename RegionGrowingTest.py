import numpy as np
import cv2

from RegionGrowing import RegionGrowing

imageFile = 'Lines.jpg'

grayScaleImg = cv2.imread(imageFile, 0)
coloredImg = cv2.imread(imageFile)

segmentedImg = RegionGrowing(grayScaleImg, threshold = 15)

def onClick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Adding a new region...")
        label = segmentedImg.addRegion((y, x))
        coloredImg[segmentedImg.labelImg == label] = np.array(np.random.choice(255, 3))
        cv2.imshow('image', coloredImg)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('image', onClick)
cv2.imshow('image', coloredImg)

cv2.waitKey(0)
cv2.destroyAllWindows()
