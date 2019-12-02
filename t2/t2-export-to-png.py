import numpy as np
import cv2
from math import exp

glb = 0
bern = 0
nib = 0
sauv = 0
pms_c = 0
contr = 0
med = 0
mediana_c = 0

def global_threshold(path,t):
    global glb
    img = cv2.imread(path, -1)
    r = np.where(img > t, 255, 0)
    cv2.imwrite('global/o-glob-'+str(glb)+'-'+str(t)+'.png',r)
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
    
    
    cv2.imwrite('bernsen/o-ber-'+str(bern)+'-'+str(ws*2+1)+'-'+l+'.png',
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
    cv2.imwrite('niblack/o-nib-'+str(nib)+'-'+str(ws*2+1)+'-'+str(k)+'.png',
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
    cv2.imwrite('sauvola/o-sauv-'+str(sauv)+'-'+str(ws*2+1)+'-'+str(k)+'-'+str(R)+'.png',
                                                                result.astype(np.uint8) )
    return result
    
def pms(path, ws, k=0.25, R=0.5, p=2, q=10):
    global pms

    img = cv2.imread(path,-1)
    img = img.astype(np.float)/255.0
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
    
    cv2.imwrite('pms/o-pms-'+str(pms_c)+'-'+str(ws*2+1)+'-'+str(k).replace(',','')
                                 +'-'+str(R).replace(',','')+'-'+str(p)+'-'+str(q)
                                 +'.png',result.astype(np.uint8) )
    return result

def contraste(path, ws, mt=False):
    global contr

    img = cv2.imread(path,-1)
    result = np.zeros_like(img)
    ws = int(ws/2)
    #print('window size: ',ws)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):        
            n = img[max(y-ws,0):y+ws+1,max(x-ws,0):x+ws+1]
            dmax = abs(n.max().astype(np.int) - img[y,x].astype(np.int))
            dmin = abs(n.min().astype(np.int) - img[y,x].astype(np.int))
            
            if mt:
                l = 'mt'
                if dmin > dmax:
                    result[y,x] = 255
                else:
                    result[y,x] = 0
            else:
                l = 'meq'
                if dmin >= dmax:
                    result[y,x] = 255
                else:
                    result[y,x] = 0
    cv2.imwrite('contraste/o-contr-'+str(contr)+'-'+str(ws*2+1)+'-'+l+'.png',result.astype(np.uint8) )
    return result


def media(path, ws, mt=False):
    global med

    img = cv2.imread(path,-1)
    result = np.zeros_like(img)
    ws = int(ws/2)
    #print('window size: ',ws)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):        
            n = img[max(y-ws,0):y+ws+1,max(x-ws,0):x+ws+1]            
            if mt:
                l = 'mt'
                if img[y,x] > np.mean(n):
                    result[y,x] = 255
                else:
                    result[y,x] = 0
            else:
                l = 'meq'
                if img[y,x] >= np.mean(n):
                    result[y,x] = 255
                else:
                    result[y,x] = 0
    cv2.imwrite('media/o-med-'+str(med)+'-'+str(ws*2+1)+'-'+l+'.png',result.astype(np.uint8) )
    return result

def mediana(path, ws, mt=False):
    global mediana_c

    img = cv2.imread(path,-1)
    result = np.zeros_like(img)
    ws = int(ws/2)
    #print('window size: ',ws)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):        
            n = img[max(y-ws,0):y+ws+1,max(x-ws,0):x+ws+1]            
            if mt:
                l = 'mt'
                if img[y,x] > np.median(n):
                    result[y,x] = 255
                else:
                    result[y,x] = 0
            else:
                l = 'meq'
                if img[y,x] >= np.median(n):                    
                    result[y,x] = 255
                else:
                    result[y,x] = 0
    cv2.imwrite('mediana/o-med-'+str(mediana_c)+'-'+str(ws*2+1)+'-'+l+
                                      '.png',result.astype(np.uint8) )
    return result


def run(func, path):
    if func == 'bernsen':
        global bern
        bernsen(path, 3)
        bernsen(path, 5)
        bernsen(path, 7)
        bernsen(path, 11)
        bernsen(path, 25)
        bernsen(path, 3 , mt=True)
        bernsen(path, 5 , mt=True)
        bernsen(path, 7 , mt=True)
        bernsen(path, 11, mt=True)
        bernsen(path, 25, mt=True)
        bern += 1
    elif func == 'niblack':
        global nib
        niblack(path, 3)
        niblack(path, 5)
        niblack(path, 7)
        niblack(path, 11)
        niblack(path, 25)
        niblack(path, 3 ,k=0.25)
        niblack(path, 5 ,k=0.25)
        niblack(path, 7 ,k=0.25)
        niblack(path, 11,k=0.25)
        niblack(path, 25,k=0.25)
        niblack(path, 3 ,k=0.75)
        niblack(path, 5 ,k=0.75)
        niblack(path, 7 ,k=0.75)
        niblack(path, 11,k=0.75)
        niblack(path, 25,k=0.75)
        nib += 1
    elif func == 'sauvola':
        global sauv
        sauvola(path, 3)
        sauvola(path, 5)
        sauvola(path, 7)
        sauvola(path, 11)
        sauvola(path, 25)
        sauvola(path, 3 ,k=0.25)
        sauvola(path, 5 ,k=0.25)
        sauvola(path, 7 ,k=0.25)
        sauvola(path, 11,k=0.25)
        sauvola(path, 25,k=0.25)
        sauvola(path, 3 ,k=0.75)
        sauvola(path, 5 ,k=0.75)
        sauvola(path, 7 ,k=0.75)
        sauvola(path, 11,k=0.75)
        sauvola(path, 25,k=0.75)
        sauv += 1
    elif func == 'pms':
        global pms_c
        pms(path, 3)
        pms(path, 5)
        pms(path, 7)
        pms(path, 11)
        pms(path, 25)
        pms(path, 3 ,k=0.25)
        pms(path, 5 ,k=0.25)
        pms(path, 7 ,k=0.25)
        pms(path, 11,k=0.25)
        pms(path, 25,k=0.25)
        pms(path, 3 ,k=0.75, p=1)
        pms(path, 5 ,k=0.75, p=1)
        pms(path, 7 ,k=0.75, p=1)
        pms(path, 11,k=0.75, p=1)
        pms(path, 25,k=0.75, p=1)
        pms(path, 3 ,k=0.25, p=3)
        pms(path, 5 ,k=0.25, p=3)
        pms(path, 7 ,k=0.25, p=3)
        pms(path, 11,k=0.25, p=3)
        pms(path, 25,k=0.25, p=3)
        pms(path, 3 ,k=0.75, q = 5)
        pms(path, 5 ,k=0.75, q = 5)
        pms(path, 7 ,k=0.75, q = 5)
        pms(path, 11,k=0.75, q = 5)
        pms(path, 25,k=0.75, q = 5)
        pms(path, 3 ,k=0.25, q = 15)
        pms(path, 5 ,k=0.25, q = 15)
        pms(path, 7 ,k=0.25, q = 15)
        pms(path, 11,k=0.25, q = 15)
        pms(path, 25,k=0.25, q = 15)
        pms_c += 1
    elif func == 'contraste':
        global contr
        contraste(path, 3)
        contraste(path, 5)
        contraste(path, 7)
        contraste(path, 11)
        contraste(path, 25)
        contraste(path, 3 , mt=True)
        contraste(path, 5 , mt=True)
        contraste(path, 7 , mt=True)
        contraste(path, 11, mt=True)
        contraste(path, 25, mt=True)
        contr += 1
    elif func == 'media':
        global med
        media(path, 3)
        media(path, 5)
        media(path, 7)
        media(path, 11)
        media(path, 25)
        media(path, 3 , mt=True)
        media(path, 5 , mt=True)
        media(path, 7 , mt=True)
        media(path, 11, mt=True)
        media(path, 25, mt=True)
        med += 1
    elif func == 'mediana':
        global mediana_c
        mediana(path, 3)
        mediana(path, 5)
        mediana(path, 7)
        mediana(path, 11)
        mediana(path, 25)
        mediana(path, 3 , mt=True)
        mediana(path, 5 , mt=True)
        mediana(path, 7 , mt=True)
        mediana(path, 11, mt=True)
        mediana(path, 25, mt=True)
        mediana_c += 1
    elif func == 'global_threshold':
        global glb
        global_threshold(path, 255*0.5)
        global_threshold(path, 255*0.25)
        global_threshold(path, 255*0.33)
        global_threshold(path, 255*0.66)
        global_threshold(path, 255*0.75)
        glb+=1




paths = ['baboon.pgm','fiducial.pgm','peppers.pgm','retina.pgm','sonnet.pgm','wedge.pgm']
for path in paths:
    print('\n\nprocessing image ',path)
    print('processing global for ',path)
    run('global_threshold', path)
    print('processing bernsen for ',path)
    run('bernsen', path)
    print('processing niblack for ',path)
    run('niblack', path)
    print('processing sauvola for ',path)
    run('sauvola', path)
    print('processing pms for ',path)
    run('pms', path)
    print('processing contraste for ',path)
    run('contraste', path)
    print('processing media for ',path)
    run('media', path)
    print('processing mediana for ',path)
    run('mediana', path)













