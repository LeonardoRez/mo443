import numpy as np
import cv2 as cv

#defining masks 
h1 = np.array(	[
					[0,0,-1,0,0],
					[0,-1,-2,-1,0],	
					[-1,-2,16,-2,-1],
					[0,-1, -2 ,-1,0],	
					[ 0 ,0,-1,0,0],
				]	
			)
h2 = np.array(	[
					[1,4,6,4,1],
					[4,16,24,16,4],
					[6,24,36,24,6],
					[4,16,24,16,4],	
					[1,4,6,4,1],
				]	
			)/256
h3 = np.array(	[
					[-1,0,1],
					[-2,0,2],
					[-1,0,1],
				]	
			)
h4 = np.array(	[
					[-1,-2,-1],
					[0,0,0],
					[1,2,1],
				]	
			)


img = cv.imread('baboon.png',cv.IMREAD_GRAYSCALE)

if img is None:
	raise FileNotFoundError('Image not found')
else:	
	cv.imshow('teste',img)
	cv.waitKey(0)
	cv.destroyAllWindows()

