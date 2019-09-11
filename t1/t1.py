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
                        img[y+ky, x+kx] = cut_limits( img[y+ky, x+kx] + kernel[ky, kx+x_step]*error )

    return result.astype(np.uint8)*255


def test(img_path, kernel, label):
    img = cv2.imread(img_path)
    img[:,:,0] = error_diffusion(img[:,:,0], kernel)
    img[:,:,1] = error_diffusion(img[:,:,1], kernel)
    img[:,:,2] = error_diffusion(img[:,:,2], kernel)
    cv2.imwrite('o-'+label+'-1.png', img)
    
    img = cv2.imread(img_path)
    img[:,:,0] = error_diffusion(img[:,:,0], kernel, alternate=True)
    img[:,:,1] = error_diffusion(img[:,:,1], kernel, alternate=True)
    img[:,:,2] = error_diffusion(img[:,:,2], kernel, alternate=True)
    cv2.imwrite('o-'+label+'-2.png', img)


########################## floyd and steinberg tests ##########################
# preparing kernel
kernel = np.array([ [0,0,7],
                    [3,5,1]])
kernel = kernel/kernel.sum()

test('a-baboon_colored.png', kernel, 'fs')
test('a-monalisa_colored.png', kernel, 'fs')
test('a-peppers_colored.png', kernel, 'fs')
test('a-watch_colored.png', kernel, 'fs')


########################## stevenson and arce tests ##########################
kernel = np.array([ [0,0,0,0,0,32,0],
                    [12,0,26,0,30,0,16],
                    [0,12,0,26,0,12,0],
                    [5,0,12,0,12,0,5] ])
kernel = kernel/kernel.sum()


test('a-baboon_colored.png', kernel, 'sa')
test('a-monalisa_colored.png', kernel, 'sa')
test('a-peppers_colored.png', kernel, 'sa')
test('a-watch_colored.png', kernel, 'sa')


########################## burkes tests  ##########################
kernel = np.array([ [0,0,0,8,4],
                    [2,4,8,4,2] ])
kernel = kernel/kernel.sum()

test('a-baboon_colored.png', kernel, 'b')
test('a-monalisa_colored.png', kernel, 'b')
test('a-peppers_colored.png', kernel, 'b')
test('a-watch_colored.png', kernel, 'b')


########################## Sierra tests  ##########################
kernel = np.array([ [0,0,0,5,3],
                    [2,4,5,4,2],
                    [0,2,3,2,0] ])
kernel = kernel/kernel.sum()

test('a-baboon_colored.png', kernel, 'si')
test('a-monalisa_colored.png', kernel, 'si')
test('a-peppers_colored.png', kernel, 'si')
test('a-watch_colored.png', kernel, 'si')


########################## Stucki tests  ##########################
kernel = np.array([ [0,0,0,8,4],
                    [2,4,8,4,2],
                    [1,2,4,2,1] ])
kernel = kernel/kernel.sum()

test('a-baboon_colored.png', kernel, 'st')
test('a-monalisa_colored.png', kernel, 'st')
test('a-peppers_colored.png', kernel, 'st')
test('a-watch_colored.png', kernel, 'st')

 
########################## Jarvis, Judice e Ninke tests  ##########################
kernel = np.array([ [0,0,0,7,5],
                    [3,5,7,5,3],
                    [1,3,5,3,1] ])
kernel = kernel/kernel.sum()

test('a-baboon_colored.png', kernel, 'jjn')
test('a-monalisa_colored.png', kernel, 'jjn')
test('a-peppers_colored.png', kernel, 'jjn')
test('a-watch_colored.png', kernel, 'jjn')


