import numpy as np
import cv2

def floyd_steinberg(img, kernel):    
    result = np.zeros(img.shape)
    h,w = img.shape
    for y in range(h):
        for x in range(w):
            if img[y,x] > 128:
                result[y,x] = 1
            error = img[y,x] - result[y,x]*255

            kh = kernel.shape[0]
            kw = kernel.shape[1]
            x_step = int(kw//2)

            for ky in range(kh):
                for kx in range((-x_step),(x_step),1):
                    if ((y+ky) < h) and ((x+kx) >= 0) and ((x+kx) < w):
                        img[y+ky, x+kx] += kernel[ky, kx+x_step]*error

    return result.astype(np.uint8)*255

####### floyd and steinberg tests #######
kernel = np.array([ [0,0,7],
                    [3,5,1]])
kernel = kernel/kernel.sum()
#monochrome test
img = cv2.imread('baboon.png', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('fs-1-1.png', floyd_steinberg(img.copy(), kernel))

#rgb test
img = cv2.imread('baboon_colored.png')
img[:,:,0] = floyd_steinberg(img[:,:,0], kernel)
img[:,:,1] = floyd_steinberg(img[:,:,1], kernel)
img[:,:,2] = floyd_steinberg(img[:,:,2], kernel)
cv2.imwrite('fs-1-2.png', img)

####### stevenson and arce tests #######
kernel = np.array([ [0,0,0,0,0,32,0],
                    [12,0,26,0,30,0,16],
                    [0,12,0,26,0,12,0],
                    [5,0,12,0,12,0,5] ])
kernel = kernel/kernel.sum()
#monochrome test
img = cv2.imread('baboon.png', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('fs-2-1.png', floyd_steinberg(img.copy(), kernel))

#rgb test
img = cv2.imread('baboon_colored.png')
img[:,:,0] = floyd_steinberg(img[:,:,0], kernel)
img[:,:,1] = floyd_steinberg(img[:,:,1], kernel)
img[:,:,2] = floyd_steinberg(img[:,:,2], kernel)
cv2.imwrite('fs-2-2.png', img)
