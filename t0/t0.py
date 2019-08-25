import numpy as np
import cv2

def negative(img):
	result = (np.ones(img.shape)*255)-img
	cv2.imwrite('1-i.png', result)


def change_interval(img,f,g):
    a = ( g[1] - g[0] )/( f[1] - f[0] )
    result = (img*a).astype(np.uint8) + g[0]
    cv2.imwrite('1-ii.png',result)


def bright(img, gamma,label):
    result = (img/255)**(1/gamma)
    result = (result*255).astype(np.uint8)
    cv2.imwrite('2-'+label+'.png',result)


def plain_bit(img, plain):
    result = np.where(img & (2**plain)>0, 255, 0)
    cv2.imwrite('3-'+str(plain)+'.png',result)


def mosaic(img,order):
    result = img.copy()

    sz = (len(order)**(1/2)) #get mosaic dimentions based on order lengh

    #get width and height of mosaic blocks
    h, w = img.shape
    bw = int(w//sz)
    bh = int(h//sz)

    #calculate anchor points based on image dimentions and block dimentions
    a = [(y,x) for y in range(0,int(h),int(bh)) for x in range(0,int(w),int(bw))] #anchor pixels

    #normalize order to 0, ... ,n
    order = np.array(order) - 1
    

    for i in range(len(order)):
        #populate new image with chunks extracted from the original
        result[ a[i][0] : a[i][0]+bh , a[i][1] : a[i][1]+bw ] = img[ a[order[i]][0] : a[order[i]][0]+bh , a[order[i]][1] : a[order[i]][1]+bw ]
    
    cv2.imwrite('4.png',result)


def linear_blend(img1, img2, alpha):
    a1 = 1-alpha
    a2 = alpha
    
    img1 = img1.copy()/255
    img2 = img2.copy()/255

    result = a1*img1 + a2*img2
    
    result = (result*255).astype(np.uint8)

    cv2.imwrite('5-'+str(alpha).split('.')[1:][0]+'.png',result)










img = cv2.imread("baboon.png", cv2.IMREAD_GRAYSCALE)	
negative(img) #questao 1 i
change_interval(img,(0,255),(100,200)) #questao 1 ii
bright(img,1.5,'i') #questao 2 i
bright(img,2.5,'ii') #questao 2 ii
bright(img,3.5,'iii') #questao 2 iii
for i in range(8):
    plain_bit(img,i)
mosaic(img,[6,11,13,3,8,16,1,9,12,14,2,7,4,15,10,5])
img2 = cv2.imread("butterfly.png", cv2.IMREAD_GRAYSCALE)	
linear_blend(img,img2,0.8)
linear_blend(img,img2,0.5)
linear_blend(img,img2,0.2)
