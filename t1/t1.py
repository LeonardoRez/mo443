import numpy as np
import cv2

def error_diffusion(img, kernel, alternate=False):    
    result = np.zeros(img.shape)
    k = kernel.copy()
    fk = k[:,::-1]
    h,w = img.shape
    for y in range(h):

        if alternate and y%2>0:
            range_w = range(w-1, -1,-1)
            kernel = fk
        else:
            range_w = range(w)
            kernel = k

        for x in range_w:
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

########################## floyd and steinberg tests ##########################
kernel = np.array([ [0,0,7],
                    [3,5,1]])
kernel = kernel/kernel.sum()
#monochrome test
img = cv2.imread('baboon.png', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('fs-1-1-1.png', error_diffusion(img.copy(), kernel))
cv2.imwrite('fs-1-1-2.png', error_diffusion(img.copy(), kernel, alternate=True))

#rgb test
img = cv2.imread('baboon_colored.png')
img[:,:,0] = error_diffusion(img[:,:,0], kernel)
img[:,:,1] = error_diffusion(img[:,:,1], kernel)
img[:,:,2] = error_diffusion(img[:,:,2], kernel)
cv2.imwrite('fs-1-2-1.png', img)

img = cv2.imread('baboon_colored.png')
img[:,:,0] = error_diffusion(img[:,:,0], kernel, alternate=True)
img[:,:,1] = error_diffusion(img[:,:,1], kernel, alternate=True)
img[:,:,2] = error_diffusion(img[:,:,2], kernel, alternate=True)
cv2.imwrite('fs-1-2-2.png', img)


# ########################## stevenson and arce tests ##########################
# kernel = np.array([ [0,0,0,0,0,32,0],
#                     [12,0,26,0,30,0,16],
#                     [0,12,0,26,0,12,0],
#                     [5,0,12,0,12,0,5] ])
# kernel = kernel/kernel.sum()
# #monochrome test
# img = cv2.imread('baboon.png', cv2.IMREAD_GRAYSCALE)
# cv2.imwrite('sa-1-1-1.png', error_diffusion(img.copy(), kernel))
# cv2.imwrite('sa-1-1-2.png', error_diffusion(img.copy(), kernel, alternate=True))
# 
# #rgb test
# img = cv2.imread('baboon_colored.png')
# img[:,:,0] = error_diffusion(img[:,:,0], kernel)
# img[:,:,1] = error_diffusion(img[:,:,1], kernel)
# img[:,:,2] = error_diffusion(img[:,:,2], kernel)
# cv2.imwrite('sa-1-2-1.png', img)
# 
# img = cv2.imread('baboon_colored.png')
# img[:,:,0] = error_diffusion(img[:,:,0], kernel, alternate=True)
# img[:,:,1] = error_diffusion(img[:,:,1], kernel, alternate=True)
# img[:,:,2] = error_diffusion(img[:,:,2], kernel, alternate=True)
# cv2.imwrite('sa-1-2-2.png', img)
#
# 
# ########################## burkes tests  ##########################
# kernel = np.array([ [0,0,0,8,4],
#                     [2,4,8,4,2] ])
# kernel = kernel/kernel.sum()
# #monochrome test
# img = cv2.imread('baboon.png', cv2.IMREAD_GRAYSCALE)
# cv2.imwrite('b-1-1-1.png', error_diffusion(img.copy(), kernel))
# cv2.imwrite('b-1-1-2.png', error_diffusion(img.copy(), kernel, alternate=True))
# 
# #rgb test
# img = cv2.imread('baboon_colored.png')
# img[:,:,0] = error_diffusion(img[:,:,0], kernel)
# img[:,:,1] = error_diffusion(img[:,:,1], kernel)
# img[:,:,2] = error_diffusion(img[:,:,2], kernel)
# cv2.imwrite('b-1-2-1.png', img)
# 
# img = cv2.imread('baboon_colored.png')
# img[:,:,0] = error_diffusion(img[:,:,0], kernel, alternate=True)
# img[:,:,1] = error_diffusion(img[:,:,1], kernel, alternate=True)
# img[:,:,2] = error_diffusion(img[:,:,2], kernel, alternate=True)
# cv2.imwrite('b-1-2-2.png', img)
#
# 
# ########################## Sierra tests  ##########################
# kernel = np.array([ [0,0,0,5,3],
#                     [2,4,5,4,2],
#                     [0,2,3,2,0] ])
# kernel = kernel/kernel.sum()
# #monochrome test
# img = cv2.imread('baboon.png', cv2.IMREAD_GRAYSCALE)
# cv2.imwrite('si-1-1-1.png', error_diffusion(img.copy(), kernel))
# cv2.imwrite('si-1-1-2.png', error_diffusion(img.copy(), kernel, alternate=True))
# 
# #rgb test
# img = cv2.imread('baboon_colored.png')
# img[:,:,0] = error_diffusion(img[:,:,0], kernel)
# img[:,:,1] = error_diffusion(img[:,:,1], kernel)
# img[:,:,2] = error_diffusion(img[:,:,2], kernel)
# cv2.imwrite('si-1-2-1.png', img)
# 
# img = cv2.imread('baboon_colored.png')
# img[:,:,0] = error_diffusion(img[:,:,0], kernel, alternate=True)
# img[:,:,1] = error_diffusion(img[:,:,1], kernel, alternate=True)
# img[:,:,2] = error_diffusion(img[:,:,2], kernel, alternate=True)
# cv2.imwrite('si-1-2-2.png', img)
# 
# 
# ########################## Stucki tests  ##########################
# kernel = np.array([ [0,0,0,8,4],
#                     [2,4,8,4,2],
#                     [1,2,4,2,1] ])
# kernel = kernel/kernel.sum()
# #monochrome test
# img = cv2.imread('baboon.png', cv2.IMREAD_GRAYSCALE)
# cv2.imwrite('st-1-1-1.png', error_diffusion(img.copy(), kernel))
# cv2.imwrite('st-1-1-2.png', error_diffusion(img.copy(), kernel, alternate=True))
# 
# #rgb test
# img = cv2.imread('baboon_colored.png')
# img[:,:,0] = error_diffusion(img[:,:,0], kernel)
# img[:,:,1] = error_diffusion(img[:,:,1], kernel)
# img[:,:,2] = error_diffusion(img[:,:,2], kernel)
# cv2.imwrite('st-1-2-1.png', img)
# 
# img = cv2.imread('baboon_colored.png')
# img[:,:,0] = error_diffusion(img[:,:,0], kernel, alternate=True)
# img[:,:,1] = error_diffusion(img[:,:,1], kernel, alternate=True)
# img[:,:,2] = error_diffusion(img[:,:,2], kernel, alternate=True)
# cv2.imwrite('st-1-2-2.png', img)
# 
#  
# ########################## Sierra tests  ##########################
# kernel = np.array([ [0,0,0,7,5],
#                     [3,5,7,5,3],
#                     [1,3,5,3,1] ])
# kernel = kernel/kernel.sum()
# #monochrome test
# img = cv2.imread('baboon.png', cv2.IMREAD_GRAYSCALE)
# cv2.imwrite('jjn-1-1-1.png', error_diffusion(img.copy(), kernel))
# cv2.imwrite('jjn-1-1-2.png', error_diffusion(img.copy(), kernel, alternate=True))
# 
# #rgb test
# img = cv2.imread('baboon_colored.png')
# img[:,:,0] = error_diffusion(img[:,:,0], kernel)
# img[:,:,1] = error_diffusion(img[:,:,1], kernel)
# img[:,:,2] = error_diffusion(img[:,:,2], kernel)
# cv2.imwrite('jjn-1-2-1.png', img)
# 
# img = cv2.imread('baboon_colored.png')
# img[:,:,0] = error_diffusion(img[:,:,0], kernel, alternate=True)
# img[:,:,1] = error_diffusion(img[:,:,1], kernel, alternate=True)
# img[:,:,2] = error_diffusion(img[:,:,2], kernel, alternate=True)
# cv2.imwrite('jjn-1-2-2.png', img)




