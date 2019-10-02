import numpy as np
import cv2
from math import exp
from matplotlib import pyplot as plt


#image counters
glb = 0
bern = 0
nib = 0
sauv = 0
pms_c = 0
contr = 0
med = 0
mediana_c = 0


#plot function
def plot(img, result_path):
    plt.clf()
    plt.hist(img.ravel(),256,[0,256])
    plt.xlabel('Bright')
    plt.xlabel('Quantity')
    plt.savefig('histograms/'+result_path[:-3]+'png')



#threshold functions
def global_threshold(path,t):
    global glb
    img = cv2.imread(path, -1)
    result = np.where(img > t, 255, 0)

    result_path = 'global/o-glob-'+str(glb)+'-'+str(t)+'.pgm' 
    cv2.imwrite( result_path,result.astype(np.uint8) )
    cv2.imwrite( result_path[:-3]+'png',result.astype(np.uint8) )
    plot(result, result_path.split('/')[1])    

    return ( (result/255).sum() / (result.shape[0]*result.shape[1]) )




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
    
    result_path = 'bernsen/o-ber-'+str(bern)+'-'+str(ws*2+1)+'-'+l+'.pgm' 
    cv2.imwrite( result_path,result.astype(np.uint8) )
    cv2.imwrite( result_path[:-3]+'png',result.astype(np.uint8) )
    plot(result, result_path.split('/')[1])    

    #return black pixel proportion
    return ( (result/255).sum() / (result.shape[0]*result.shape[1]) )

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
    result_path = 'niblack/o-nib-'+str(nib)+'-'+str(ws*2+1)+'-'+str(k).replace('.','')+'.pgm'
    cv2.imwrite( result_path,result.astype(np.uint8) )
    cv2.imwrite( result_path[:-3]+'png',result.astype(np.uint8) )
    plot(result, result_path.split('/')[1])    

    return ( (result/255).sum() / (result.shape[0]*result.shape[1]) )

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
    result_path = 'sauvola/o-sauv-'+str(sauv)+'-'+str(ws*2+1)+'-'+str(k).replace('.','')+'-'+str(R)+'.pgm'
    cv2.imwrite( result_path,result.astype(np.uint8) )
    cv2.imwrite( result_path[:-3]+'png',result.astype(np.uint8) )
    plot(result, result_path.split('/')[1])    

    return ( (result/255).sum() / (result.shape[0]*result.shape[1]) )
    
def pms(path, ws, k=0.25, R=0.5, p=2, q=10):
    
    global pms
    ws = int(ws/2)

    result_path = 'pms/o-pms-'+str(pms_c)+'-'+str(ws*2+1)+'-'+str(k).replace('.','')+'-'+str(R).replace('.','')+'-'+str(p)+'-'+str(q)+'.pgm'
    print('processing pms ',result_path[4:])
    img = cv2.imread(path,-1)
    img = img.astype(np.float)/255.0
    result = np.zeros_like(img)
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
    
    cv2.imwrite( result_path,result.astype(np.uint8) )
    cv2.imwrite( result_path[:-3]+'png',result.astype(np.uint8) )
    plot(result, result_path.split('/')[1])    

    return ( (result/255).sum() / (result.shape[0]*result.shape[1]) )

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
    result_path = 'contraste/o-contr-'+str(contr)+'-'+str(ws*2+1)+'-'+l+'.pgm'
    cv2.imwrite( result_path,result.astype(np.uint8) )
    cv2.imwrite( result_path[:-3]+'png',result.astype(np.uint8) )
    plot(result, result_path.split('/')[1])    

    return ( (result/255).sum() / (result.shape[0]*result.shape[1]) )


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

    result_path = 'media/o-med-'+str(med)+'-'+str(ws*2+1)+'-'+l+'.pgm'
    cv2.imwrite( result_path,result.astype(np.uint8) )
    cv2.imwrite( result_path[:-3]+'png',result.astype(np.uint8) )
    plot(result, result_path.split('/')[1])    

    return ( (result/255).sum() / (result.shape[0]*result.shape[1]) )

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

    result_path = 'mediana/o-med-'+str(mediana_c)+'-'+str(ws*2+1)+'-'+l+'.pgm'
    cv2.imwrite( result_path,result.astype(np.uint8) )
    cv2.imwrite( result_path[:-3]+'png',result.astype(np.uint8) )
    plot(result, result_path.split('/')[1])    

    return ( (result/255).sum() / (result.shape[0]*result.shape[1]) )


def run(func, path):
    if func == 'bernsen':
        global bern        
        print('\tbernsen: ', bernsen(path, 3))
        print('\tbernsen: ', bernsen(path, 5))
        print('\tbernsen: ', bernsen(path, 7))
        print('\tbernsen: ', bernsen(path, 11))
        print('\tbernsen: ', bernsen(path, 25))
        print('\tbernsen: ', bernsen(path, 3 , mt=True))
        print('\tbernsen: ', bernsen(path, 5 , mt=True))
        print('\tbernsen: ', bernsen(path, 7 , mt=True))
        print('\tbernsen: ', bernsen(path, 11, mt=True))
        print('\tbernsen: ', bernsen(path, 25, mt=True))
        bern += 1
    elif func == 'niblack':
        global nib
        print('\tniblack: ', niblack(path, 3        ))
        print('\tniblack: ', niblack(path, 5        ))
        print('\tniblack: ', niblack(path, 7        ))
        print('\tniblack: ', niblack(path, 11       ))
        print('\tniblack: ', niblack(path, 25       ))
        print('\tniblack: ', niblack(path, 3 ,k=0.25))
        print('\tniblack: ', niblack(path, 5 ,k=0.25))
        print('\tniblack: ', niblack(path, 7 ,k=0.25))
        print('\tniblack: ', niblack(path, 11,k=0.25))
        print('\tniblack: ', niblack(path, 25,k=0.25))
        print('\tniblack: ', niblack(path, 3 ,k=0.75))
        print('\tniblack: ', niblack(path, 5 ,k=0.75))
        print('\tniblack: ', niblack(path, 7 ,k=0.75))
        print('\tniblack: ', niblack(path, 11,k=0.75))
        print('\tniblack: ', niblack(path, 25,k=0.75))
        nib += 1
    elif func == 'sauvola':
        global sauv
        print('\tsauvola: ', sauvola(path, 3         ))
        print('\tsauvola: ', sauvola(path, 5         ))
        print('\tsauvola: ', sauvola(path, 7         ))
        print('\tsauvola: ', sauvola(path, 11        ))
        print('\tsauvola: ', sauvola(path, 25        ))
        print('\tsauvola: ', sauvola(path, 3 ,k=0.25 ))
        print('\tsauvola: ', sauvola(path, 5 ,k=0.25 ))
        print('\tsauvola: ', sauvola(path, 7 ,k=0.25 ))
        print('\tsauvola: ', sauvola(path, 11,k=0.25 ))
        print('\tsauvola: ', sauvola(path, 25,k=0.25 ))
        print('\tsauvola: ', sauvola(path, 3 ,k=0.75 ))
        print('\tsauvola: ', sauvola(path, 5 ,k=0.75 ))
        print('\tsauvola: ', sauvola(path, 7 ,k=0.75 ))
        print('\tsauvola: ', sauvola(path, 11,k=0.75 ))
        print('\tsauvola: ', sauvola(path, 25,k=0.75 ))
        sauv += 1
    elif func == 'pms':
        global pms_c
        print('\tpms: ', pms(path, 3                 ))
        print('\tpms: ', pms(path, 5                 ))
        print('\tpms: ', pms(path, 7                 ))
        print('\tpms: ', pms(path, 11                ))
        print('\tpms: ', pms(path, 25                ))
        print('\tpms: ', pms(path, 3 ,k=0.125        ))
        print('\tpms: ', pms(path, 5 ,k=0.125        ))
        print('\tpms: ', pms(path, 7 ,k=0.125        ))
        print('\tpms: ', pms(path, 11,k=0.125        ))
        print('\tpms: ', pms(path, 25,k=0.125        ))
        print('\tpms: ', pms(path, 3 ,k=0.5, p=1     ))
        print('\tpms: ', pms(path, 5 ,k=0.5, p=1     ))
        print('\tpms: ', pms(path, 7 ,k=0.5, p=1     ))
        print('\tpms: ', pms(path, 11,k=0.5, p=1     ))
        print('\tpms: ', pms(path, 25,k=0.5, p=1     ))
        print('\tpms: ', pms(path, 3 ,k=0.125, p=3   ))
        print('\tpms: ', pms(path, 5 ,k=0.125, p=3   ))
        print('\tpms: ', pms(path, 7 ,k=0.125, p=3   ))
        print('\tpms: ', pms(path, 11,k=0.125, p=3   ))
        print('\tpms: ', pms(path, 25,k=0.125, p=3   ))
        print('\tpms: ', pms(path, 3 ,k=0.5, q = 5   ))
        print('\tpms: ', pms(path, 5 ,k=0.5, q = 5   ))
        print('\tpms: ', pms(path, 7 ,k=0.5, q = 5   ))
        print('\tpms: ', pms(path, 11,k=0.5, q = 5   ))
        print('\tpms: ', pms(path, 25,k=0.5, q = 5   ))
        print('\tpms: ', pms(path, 3 ,k=0.125, q = 15))
        print('\tpms: ', pms(path, 5 ,k=0.125, q = 15))
        print('\tpms: ', pms(path, 7 ,k=0.125, q = 15))
        print('\tpms: ', pms(path, 11,k=0.125, q = 15))
        print('\tpms: ', pms(path, 25,k=0.125, q = 15))
        pms_c += 1
    elif func == 'contraste':
        global contr
        print('\tcontraste: ', contraste(path, 3          ))
        print('\tcontraste: ', contraste(path, 5          ))
        print('\tcontraste: ', contraste(path, 7          ))
        print('\tcontraste: ', contraste(path, 11         ))
        print('\tcontraste: ', contraste(path, 25         ))
        print('\tcontraste: ', contraste(path, 3 , mt=True))
        print('\tcontraste: ', contraste(path, 5 , mt=True))
        print('\tcontraste: ', contraste(path, 7 , mt=True))
        print('\tcontraste: ', contraste(path, 11, mt=True))
        print('\tcontraste: ', contraste(path, 25, mt=True))
        contr += 1
    elif func == 'media':
        global med
        print('\tmedia: ', media(path, 3          ))
        print('\tmedia: ', media(path, 5          ))
        print('\tmedia: ', media(path, 7          ))
        print('\tmedia: ', media(path, 11         ))
        print('\tmedia: ', media(path, 25         ))
        print('\tmedia: ', media(path, 3 , mt=True))
        print('\tmedia: ', media(path, 5 , mt=True))
        print('\tmedia: ', media(path, 7 , mt=True))
        print('\tmedia: ', media(path, 11, mt=True))
        print('\tmedia: ', media(path, 25, mt=True))
        med += 1
    elif func == 'mediana':
        global mediana_c
        print('\tmediana: ', mediana(path, 3          ))
        print('\tmediana: ', mediana(path, 5          ))
        print('\tmediana: ', mediana(path, 7          ))
        print('\tmediana: ', mediana(path, 11         ))
        print('\tmediana: ', mediana(path, 25         ))
        print('\tmediana: ', mediana(path, 3 , mt=True))
        print('\tmediana: ', mediana(path, 5 , mt=True))
        print('\tmediana: ', mediana(path, 7 , mt=True))
        print('\tmediana: ', mediana(path, 11, mt=True))
        print('\tmediana: ', mediana(path, 25, mt=True))
        mediana_c += 1
    elif func == 'global_threshold':
        global glb
        print('\tglobal_threshold: ', global_threshold(path, 255*0.5 ))
        print('\tglobal_threshold: ', global_threshold(path, 255*0.25))
        print('\tglobal_threshold: ', global_threshold(path, 255*0.33))
        print('\tglobal_threshold: ', global_threshold(path, 255*0.66))
        print('\tglobal_threshold: ', global_threshold(path, 255*0.75))
        glb+=1




paths = ['baboon.pgm','fiducial.pgm','peppers.pgm','retina.pgm','sonnet.pgm','wedge.pgm']
methods = ['global_threshold','contraste','media','mediana','pms','bernsen','niblack','sauvola']

#bp_counter = np.zeros( (len(paths), len(methods) ) 

#for path in paths:
#    print('\n\nProcessing image ', path)
#    for m in methods:
#        print('processing',m,' for ',path)
#        run(m,path)


for m in methods:
    print('\n\nProcessing method ', m)
    for path in paths:
        print('processing',m,' for ',path)
        run(m,path)










