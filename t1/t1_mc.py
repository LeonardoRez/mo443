import numpy as np
import cv2

def cut_limits(value):
    if value > 255:
        return 255
    if value < 0:
        return 0
    return value


def error_diffusion(img, kernel, alternate=False):    
    result = np.zeros(img.shape)
    k = kernel.copy()
    fk = k[:,::-1]
    h,w = img.shape

    # kernel dimentions
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    x_step = int(kw//2) # distance from the borders to the center

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

            for ky in range(kh):
                for kx in range((-x_step),(x_step),1):
                    if ((y+ky) < h) and ((x+kx) >= 0) and ((x+kx) < w):
                        img[y+ky, x+kx] = cut_limits( img[y+ky, x+kx] + kernel[ky, kx+x_step]*error )

    return result.astype(np.uint8)*255


def test(img_path, kernel, label):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('o-'+label+'-1_mc.png', error_diffusion(img, kernel))
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('o-'+label+'-2_mc.png', error_diffusion(img, kernel, alternate=True))


########################## floyd and steinberg tests ##########################
# preparing kernel
kernel = np.array([ [0,0,7],
                    [3,5,1]])
kernel = kernel/kernel.sum()

test('a-baboon_colored.png', kernel, 'fs-b')
test('a-monalisa_colored.png', kernel, 'fs-m')
test('a-peppers_colored.png', kernel, 'fs-p')
test('a-watch_colored.png', kernel, 'fs-w')


########################## stevenson and arce tests ##########################
kernel = np.array([ [0,0,0,0,0,32,0],
                    [12,0,26,0,30,0,16],
                    [0,12,0,26,0,12,0],
                    [5,0,12,0,12,0,5] ])
kernel = kernel/kernel.sum()


test('a-baboon_colored.png', kernel, 'sa-b')
test('a-monalisa_colored.png', kernel, 'sa-m')
test('a-peppers_colored.png', kernel, 'sa-p')
test('a-watch_colored.png', kernel, 'sa-w')


########################## burkes tests  ##########################
kernel = np.array([ [0,0,0,8,4],
                    [2,4,8,4,2] ])
kernel = kernel/kernel.sum()

test('a-baboon_colored.png', kernel, 'b-b')
test('a-monalisa_colored.png', kernel, 'b-m')
test('a-peppers_colored.png', kernel, 'b-p')
test('a-watch_colored.png', kernel, 'b-w')


########################## Sierra tests  ##########################
kernel = np.array([ [0,0,0,5,3],
                    [2,4,5,4,2],
                    [0,2,3,2,0] ])
kernel = kernel/kernel.sum()

test('a-baboon_colored.png', kernel, 'si-b')
test('a-monalisa_colored.png', kernel, 'si-m')
test('a-peppers_colored.png', kernel, 'si-p')
test('a-watch_colored.png', kernel, 'si-w')


########################## Stucki tests  ##########################
kernel = np.array([ [0,0,0,8,4],
                    [2,4,8,4,2],
                    [1,2,4,2,1] ])
kernel = kernel/kernel.sum()

test('a-baboon_colored.png', kernel, 'st-b')
test('a-monalisa_colored.png', kernel, 'st-m')
test('a-peppers_colored.png', kernel, 'st-p')
test('a-watch_colored.png', kernel, 'st-w')

 
########################## Jarvis, Judice e Ninke tests  ##########################
kernel = np.array([ [0,0,0,7,5],
                    [3,5,7,5,3],
                    [1,3,5,3,1] ])
kernel = kernel/kernel.sum()

test('a-baboon_colored.png', kernel, 'jjn-b')
test('a-monalisa_colored.png', kernel, 'jjn-m')
test('a-peppers_colored.png', kernel, 'jjn-p')
test('a-watch_colored.png', kernel, 'jjn-w')


