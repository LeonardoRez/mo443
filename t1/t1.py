import numpy as np
import cv2

def floyd_steinberg(img):    
    result = np.zeros(img.shape)
    h,w = img.shape
    for y in range(h):
        for x in range(w):
            if img[y,x] > 128:
                result[y,x] = 1
            error = img[y,x] - result[y,x]*255
            if y < h-1:
                img[y+1,x] += (7/16)*error
            if y > 0 and x < w-1:
                img[y-1,x+1] += (3/16)*error
            if x < w-1:
                img[y,x+1] += (5/16)*error
            if y < h-1 and x < w-1:
                img[y+1,x+1] += (1/16)*error
    return result.astype(np.uint8)*255

####### floyd and steinberg tests #######

#monochrome test
img = cv2.imread('baboon.png', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('fs-1.png', floyd_steinberg(img.copy()))

#rgb test
img = cv2.imread('baboon_colored.png')
img[:,:,0] = floyd_steinberg(img[:,:,0])
img[:,:,1] = floyd_steinberg(img[:,:,1])
img[:,:,2] = floyd_steinberg(img[:,:,2])
cv2.imwrite('fs-2.png', img)
