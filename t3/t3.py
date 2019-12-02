import numpy as np
import cv2
from plot_graph import plot_histogram
import sys


def binarize(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    result = np.where(img < 230, 255, 0)
    return result.astype('uint8')


def get_borders(img, approach):
    opt = int(approach)
    kernel = np.ones((3,3)).astype(np.uint8)
    d = cv2.dilate(img, kernel, iterations = 1)
    e = cv2.erode(img, kernel, iterations = 1)
    if opt==1:
        print('returning dilatation - image')
        return (d-img)
    elif opt==2:
        print('returning image - erosion')
        return (img-e)
    else:
        print('returning dilatation - erosion')
        return (d-e)
    
                                
def get_contours(img):
    contours, _ = cv2.findContours(img.astype('uint8'),
                                   cv2.RETR_TREE, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_moments(contours):
    m = []
    for c in contours:
        m.append(cv2.moments(c))
    return m


def get_centroids(moments):
    centroids = []

    for region_moments in moments:
        cx = int(region_moments['m10']/region_moments['m00'])
        cy = int(region_moments['m01']/region_moments['m00'])
        centroids.append([cy,cx])

    return centroids


def label_centroids(img, centroids):
    result = img.copy()
    i = 0
    for c in centroids:
        #tests if the object is dark or bright to change the label color
        gray_level = img[c[0],c[1]].sum()/3
        if gray_level > 127:
            color = (0,0,0)
        else:
            color = (255,255,255)
        cv2.putText(result,str(i), (c[1]-4,c[0]+4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color )
        i+=1

    return result

def get_areas(contours):
    areas = []
    for contour in contours:
        areas.append( cv2.contourArea(contour) )

    return np.array(areas)

def get_perimeters(contours):
    perimeters = []
    for contour in contours:
        perimeters.append( cv2.arcLength(contour,True) )

    return np.array(perimeters)

def get_eccentricities(moments):
    ecc = []
    for m in moments:
        ecc.append( ( (m['m20']-m['m02'])**2 + 4*m['m11']**2 )/(m['m20']+m['m02'])**2 )

    return ecc

def get_solidities(contour):
    solidities = []
    for c in contour:
        area = cv2.contourArea(c)
        hull = cv2.contourArea(cv2.convexHull(c))
        solidities.append((float(area)/hull))
    
    return solidities
counter=0
def main(path, output_path,opt):
    global counter
    print('----------------processing image ',path)
    original = cv2.imread(path)
    #binarizing image                                
    img = binarize(path)
    cv2.imwrite(output_path+'/threshold'+str(counter)+'.png', img)
    
    #getting the borders
    borders = get_borders(img,opt)
    cv2.imwrite(output_path+'/borders'+str(counter)+'.png', borders)
    
    #getting the centroids
    contours = get_contours(img)
    moments = get_moments(contours)
    centroids = get_centroids(moments)
    
    #label every centroid
    labeled_centroids = label_centroids(original, centroids) 
    cv2.imwrite(output_path+'/labeled_centroids'+str(counter)+'.png', labeled_centroids)
    
    #get region informations
    areas = get_areas(contours)
    perimeters = get_perimeters(contours)
    ecc = get_eccentricities(moments)
    solidities = get_solidities(contours)
    
    print('Number of regions:', len(areas))
    for i in range(len(areas)):
        print('region:{}\tarea: {}\tperimeter: {}\teccentricity: {}\t solidity: {}\n'.format(i, areas[i], perimeters[i], ecc[i], solidities[i]))
    
    #grouping regions by size
    large = 0
    avg = 0
    small = 0
    for a in areas:
        if a < 1500:
            small+=1
        elif a < 3000:
            avg+=1
        else:
            large+=1

    print('number of small regions: ', small)
    print('number of average regions: ', avg)
    print('number of large regions: ', large)
    print('\n\n')

    # plotting the histogram of regions
    plot_histogram(areas,[small,avg,large] ,output_path+'/histogram-'+str(counter)+'.png')
    counter += 1

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('missing the operation arg.\n---1 for dilatation - image\n---2 for image - erosion\n---3 for dilatation - erosion')
        print('Try to execute as python t3.py <number_listed_above>')
    else:
        opt = sys.argv[1]
        if opt == '1':
            output_p = 'only_dilatation'
        elif opt == '2':
            output_p = 'only_erosion'
        else:
            output_p='dilatation_erosion'
        main('objetos1.png',output_p, opt)
        main('objetos2.png',output_p, opt)
        main('objetos3.png',output_p, opt)














