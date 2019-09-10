from PIL import Image
import math

def brightness(img, b):
	r = Image.new("L", img.size, 0) #result image

	for l in range(img.size[0]):
		for c in range(img.size[1]):
			o = (img.getpixel((l,c))/255) #get original pixel
			o = o**(1/b) #change its value by o^(1/brightness_rate)
			r.putpixel((l,c), (int(o*255)) ) #store it on result image			
		
	r.show()
	value = str(b).replace('.','-')
	r.save('brightness_'+value+'.png')

def bit_plain(img, ordem):
	r = Image.new("L", img.size, 0)
	for l in range(img.size[0]):
		for c in range(img.size[1]): #for each pixel at original image,
			bit  = (img.getpixel((l,c)) & int(pow(2,ordem))) #check bit value, depending on order value
			r.putpixel((l,c), 255*bit) #paint it black if An = 0, 255 otherwise 

	r.show()
	r.save('bit_plain_order'+str(ordem)+'.png')


#function used in "mosaic" to get all block, based on anchor point and size (of block, in pixels) defined below 
def get_blocks(p1,p2,size):	
	c1 = [(l,c) for l in range(p1[0],p1[0]+size[0]) for c in range(p1[1],p1[1]+size[1])]
	c2 = [(l,c) for l in range(p2[0],p2[0]+size[0]) for c in range(p2[1],p2[1]+size[1])]
	return c1,c2
	


def mosaic(img,new_order, verbose=False):		
	#getting block's width and length
	size = (int(img.size[0]/4),int(img.size[1]/4))

	width =  size[0]*4	#this implies in losing 0 to 3 columns
	height =  size[1]*4 #this implies in losing 0 to 3 rows

	#changing index for interval 0-15 instead of 1-16
	new_order = [n-1 for n in new_order] 
	
	#getting the upper left pixel (anchors) of each block
	a = [ (l,c) for c in range(0,height,int(size[1])) for l in range(0,width,int(size[0])) ]
	#getting anchors in new order (no)
	no = [a[n] for n in new_order]
	if verbose:
		print("starting to populate new image")
	r = Image.new("L", img.size, 0)
	#populating new image "r" with new order of blocks
	for i in range(len(a)):
		orig_blk,new_blk = get_blocks(a[i],no[i],size) #get every pixel (by adress, not value) of each block
		for j in range(len(orig_blk)):
			r.putpixel(orig_blk[j], img.getpixel(new_blk[j])) #put pixels of the original image on new image

	if verbose:
		print("new image ready")
	r.show()
	r.save('mosaic_result.png')


def combination(img1, img2, l1, l2):
	if img1.size != img2.size:
		raise InputError('Images with different sizes')
	r = Image.new("L", img1.size, 0)

	for l in range(img1.size[0]):
		for c in range(img1.size[1]):
			new_value = int( ( img1.getpixel((l,c))*l1+img2.getpixel((l,c))*l2 )/(l1+l2) ) #calculating weighted mean
			r.putpixel((l,c), new_value )
	r.show()
	r.save('merge.png')



img = Image.open("baboon.png").convert("L") #open and convert to grayscale 
img.show()

# testing brightness as the answer 
brightness(img,1.5)
brightness(img,2.5)
brightness(img,3.5)

#testing bit plain as the answer
for i in [0,4,7]:
	bit_plain(img,i)

# #testing mosaic as the answer
new_order = [6,11,13,3,8,16,1,9,12,14,2,7,4,15,10,5] # original order is 1,2,...,16
mosaic(img, new_order,verbose=True)

#testing combination as the answer
img2 = Image.open("butterfly.png").convert("L")
img2.show()
combination(img, img2, 0.2, 0.8 )
combination(img, img2, 0.5, 0.5 )
combination(img, img2, 0.8, 0.2 )
