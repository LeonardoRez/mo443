# coding: utf-8
from PIL import Image
img = Image.open("../baboon.png").convert("L")
width =  int(img.size[0]/4)*4
height =  int(img.size[1]/4)*4
a = [ (l,c) for c in range(0,height,int(height/4)) for l in range(0,width,int(width/4)) ]
new_order = [6,11,13,3,8,16,1,9,12,14,2,7,4,15,10,5]
new_order = [n-1 for n in new_order]
no = [a[n] for n in new_order]

def get_blocks(p1,p2,size):	
	c1 = [(l,c) for l in range(p1[0]+size[0]) for c in range(p1[1]+size[1])]
	c2 = [(l,c) for l in range(p2[0]+size[0]) for c in range(p2[1]+size[1])]
	return (c1,c2)
