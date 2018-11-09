import cv2 as cv
import numpy as np
import matplotlib.image as mpimg

rgb_img = cv.imread('rostos/rostos-1.jpg', flags=cv.IMREAD_COLOR );
cv.imshow('RGB rostos',rgb_img);

hsv_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2HSV)
cv.imshow('HSV rostos',hsv_img);

lower_red = 150
upper_red = 250

lower_gre = 50
upper_gre = 200

lower_blu = 100
upper_blu = 140

height, width = hsv_img.shape[:2]

lower_white = np.array([lower_blu, lower_gre, lower_red], dtype=np.uint8)
upper_white = np.array([upper_blu, upper_gre, upper_red], dtype=np.uint8)

bin_img = cv.inRange(hsv_img, lower_white, upper_white)
cv.imshow('BIN',bin_img);

res = cv.bitwise_and(hsv_img,hsv_img, mask= bin_img)

cv.imshow('RES',res);

im_floodfill = bin_img.copy()

h, w = bin_img.shape[:2]
mask = np.zeros((h + 2, w + 2), np.uint8)

cv.floodFill(im_floodfill, mask, (0, 0), 0);

im_floodfill_inv = cv.bitwise_not(im_floodfill)
cv.imshow('FLOODFILL',im_floodfill_inv);

im_out = bin_img | im_floodfill_inv

cv.imshow('OUT',im_out);
cv.waitKey(0)
cv.destroyAllWindows()