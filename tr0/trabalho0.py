from PIL import Image
import math

def brightness(image, b):
	r = Image.new("L", image.size, 0) 	

	for l in range(image.size[0]):
		for c in range(image.size[1]):
			o = (image.getpixel((l,c))/255)
			o = o**(1/b)
			r.putpixel((l,c), (int(o*255)) )			
		
	r.show()
def bit_plain(img, ordem):
	r = Image.new("L", img.size, 0)
	for l in range(img.size[0]):
		for c in range(img.size[1]):
			bit  = (img.getpixel((l,c)) & int(pow(2,ordem)))
			# print(bit)
			if bit > 0:
				r.putpixel((l,c), 255)
			else:
				r.putpixel((l,c), 0)

	r.show()


def get_blocks(p1,p2,size):	
	c1 = [(l,c) for l in range(p1[0]+size[0]) for c in range(p1[1]+size[1])]
	c2 = [(l,c) for l in range(p2[0]+size[0]) for c in range(p2[1]+size[1])]
	return (c1,c2)
	


def mosaic(img,new_order):		
	#getting block's wigth and length
	size = (int(img.size[0]/4),int(img.size[1]/4))

	width =  size[0]*4	
	height =  size[1]*4

	#changing index for interval 0-15 instead of 1-16
	new_order = [n-1 for n in new_order] 
	
	#getting the upper left pixel (anchors) of each block
	a = [ (l,c) for c in range(0,height,int(size[1])) for l in range(0,width,int(size[0])) ]
	#getting anchors in new order (no)
	no = [a[n] for n in new_order]
	
	

	# print(a)
	# r = Image.new("L", (width,height), 0)
	
	# for i in range(len(a)/2):
	# 	for j in range(len(a)/2,len(a)):
	# 		q = get_blocks(a[i],a[j])
	# 		[r.putpixel(v, p) for p in q[0] for v in []]






img = Image.open("../baboon.png").convert("L") #abre a imagem e converte para tons de cinza
img.show()

#testing brightness as the answer 
# brightness(img,1.5)
# brightness(img,2.5)
# brightness(img,3.5)

#testing bit plain as the answer
# for i in [0,4,7]:
# 	bit_plain(img,i)

#testing mosaic as the answer
mosaic(img)