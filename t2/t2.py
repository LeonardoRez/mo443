import numpy as np
import cv2
from math import exp

glb = 0
bern = 0
nib = 0
sauv = 0
pms = 0

def global_threshold(path,t):
    global glb
    img = cv2.imread(path, -1)
    r = np.where(img > t, 255, 0)
    cv2.imwrite('o-glob-'+str(glb)+'-'+str(t)+'.pgm',r)
    return r




def bernsen(path, ws,mt=False):
    global bern

    img = cv2.imread(path,-1)
    result = np.zeros_like(img)
    ws = int(ws/2)
    #print('window size: ',ws)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):        
            n = img[max(y-ws,0):y+ws+1,max(x-ws,0):x+ws+1]
            ##print(n)
            mi = n.min()
            ma = n.max()
            v = (1.0*mi + 1.0*ma)/2
            if mt:
                l = 'mt'
                if img[y,x] > v:
                    result[y,x] = 255
                else:
                    result[y,x] = 0
            else:
                l = 'meq'
                if img[y,x] >= v:
                    result[y,x] = 255
                else:
                    result[y,x] = 0
    
    
    cv2.imwrite('bernsen/o-ber-'+str(bern)+'-'+str(ws*2+1)+'-'+l+'.pgm',
                                                result.astype(np.uint8) )
    return result

def niblack(path, ws,k=0.5):
    global nib

    img = cv2.imread(path,-1)
    result = np.zeros_like(img)
    ws = int(ws/2)
    #print('window size: ',ws)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):        
            n = img[max(y-ws,0):y+ws+1,max(x-ws,0):x+ws+1]
            ##print(n)
            v = k*np.std(n)+np.mean(n)
            if img[y,x] >= v:
                result[y,x] = 255
            else:
                result[y,x] = 0
    nib += 1
    cv2.imwrite('niblack/o-nib-'+str(nib)+'-'+str(ws*2+1)+'-'+str(k)+'.pgm',
                                                    result.astype(np.uint8) )
    return result

def sauvola(path, ws, k=0.5, R=128):
    global sauv

    img = cv2.imread(path,-1)
    result = np.zeros_like(img)
    ws = int(ws/2)
    #print('window size: ',ws)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):        
            n = img[max(y-ws,0):y+ws+1,max(x-ws,0):x+ws+1]
            ##print(n)
            v = np.mean(n) * ( 1 +  k*( np.std(n)/R - 1 ) )
            if img[y,x] >= v:
                result[y,x] = 255
            else:
                result[y,x] = 0
    sauv += 1
    cv2.imwrite('sauvola/o-sauv-'+str(sauv)+'-'+str(ws*2+1)+'-'+str(k)+'-'+str(R)+'.pgm',
                                                                result.astype(np.uint8) )
    return result
    
def pms(path, ws, k=0.25, R=0.5, p=2, q=10):
    global pms

    img = cv2.imread(path,-1)
    result = np.zeros_like(img)
    ws = int(ws/2)
    #print('window size: ',ws)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):        
            n = img[max(y-ws,0):y+ws+1,max(x-ws,0):x+ws+1]
            ##print(n)
            v = np.mean(n) * ( 1 + p*exp(-q*np.mean(n)) + k*( np.std(n)/R - 1 ) )
            if img[y,x] >= v:
                result[y,x] = 255
            else:
                result[y,x] = 0
    
    cv2.imwrite('pms/o-pms-'+str(pms)+'-'+str(ws*2+1)+'-'+str(k)+'-'+str(R).replace(',','')
                                    +'-'+str(p)+'-'+str(q)+'.pgm',result.astype(np.uint8) )
    return result





def run(func, path):
    if func == 'bernsen':
        global bern
        bernsen(path, 3)
        bernsen(path, 5)
        bernsen(path, 7)
        bernsen(path, 11)
        bernsen(path, 3 , mt=True)
        bernsen(path, 5 , mt=True)
        bernsen(path, 7 , mt=True)
        bernsen(path, 11, mt=True)
        bern += 1
    elif func == 'niblack':
        global nib
        nib = 0
        niblack(path, 3)
        niblack(path, 5)
        niblack(path, 7)
        niblack(path, 11)
        niblack(path, 3 ,k=0.25)
        niblack(path, 5 ,k=0.25)
        niblack(path, 7 ,k=0.25)
        niblack(path, 11,k=0.25)
        niblack(path, 3 ,k=0.75)
        niblack(path, 5 ,k=0.75)
        niblack(path, 7 ,k=0.75)
        niblack(path, 11,k=0.75)
    elif func == 'sauvola':
        global sauv
        sauvola(path, 3)
        sauvola(path, 5)
        sauvola(path, 7)
        sauvola(path, 11)
        sauvola(path, 3 ,k=0.25)
        sauvola(path, 5 ,k=0.25)
        sauvola(path, 7 ,k=0.25)
        sauvola(path, 11,k=0.25)
        sauvola(path, 3 ,k=0.75)
        sauvola(path, 5 ,k=0.75)
        sauvola(path, 7 ,k=0.75)
        sauvola(path, 11,k=0.75)
        sauv += 1
    elif func == 'bernsen':
        global pms
        pms = 0
        pms(path, 3)
        pms(path, 5)
        pms(path, 7)
        pms(path, 11)
        pms(path, 3 ,k=0.25)
        pms(path, 5 ,k=0.25)
        pms(path, 7 ,k=0.25)
        pms(path, 11,k=0.25)
        pms(path, 3 ,k=0.75, p=1)
        pms(path, 5 ,k=0.75, p=1)
        pms(path, 7 ,k=0.75, p=1)
        pms(path, 11,k=0.75, p=1)
        pms(path, 3 ,k=0.25, p=3)
        pms(path, 5 ,k=0.25, p=3)
        pms(path, 7 ,k=0.25, p=3)
        pms(path, 11,k=0.25, p=3)
        pms(path, 3 ,k=0.75, q = 5)
        pms(path, 5 ,k=0.75, q = 5)
        pms(path, 7 ,k=0.75, q = 5)
        pms(path, 11,k=0.75, q = 5)
        pms(path, 3 ,k=0.25, q = 15)
        pms(path, 5 ,k=0.25, q = 15)
        pms(path, 7 ,k=0.25, q = 15)
        pms(path, 11,k=0.25, q = 15)
    elif func == 'bernsen':
        bernsen(path, 3)
        bernsen(path, 5)
        bernsen(path, 7)
        bernsen(path, 11)
        bernsen(path, 3 )
        bernsen(path, 5 )
        bernsen(path, 7 )
        bernsen(path, 11)
    elif func == 'bernsen':
        bernsen(path, 3)
        bernsen(path, 5)
        bernsen(path, 7)
        bernsen(path, 11)
        bernsen(path, 3 )
        bernsen(path, 5 )
        bernsen(path, 7 )
        bernsen(path, 11)
    elif func == 'global_threshold':
        global glb
        global_threshold(path, 3)
        global_threshold(path, 5)
        global_threshold(path, 7)
        global_threshold(path, 11)
        global_threshold(path, 3 )
        global_threshold(path, 5 )
        global_threshold(path, 7 )
        global_threshold(path, 11)
        glb+=1






#run('global_threshold', 'fiducial.pgm')
#run('bernsen', 'fiducial.pgm')
#run('niblack', 'fiducial.pgm')
#run('sauvola', 'fiducial.pgm')
run('pms', 'fiducial.pgm')













