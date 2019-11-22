import numpy as np 
import cv2
import argparse
import os


#Function to compute the image with only k components
def svd(img, k):

    # start the resulting image
    A = np.zeros(img.shape)
    for i in range(img.shape[2]):
        # extract the USV from the image
        U,s,V = np.linalg.svd(img[:,:,i],full_matrices=False)
        # buld the result with just k components
        A[:,:,i] = np.dot(np.dot(U[:,:k],np.diag(s)[:k,:k]), V[:k,:])

    return A


# function to calculate the relation between the size of the input and the output
def compression_rate(i_path, o_path):
    input_size = os.stat(i_path).st_size
    output_size = os.stat(o_path).st_size
    
    return output_size/input_size


# function to calculate the root mean square error
def RMSE(i,o):
    diff = i-o
    sqrt_error = (diff**2).sum()
    # getting the h and w of the images
    h,w,_ = i.shape
    return  (sqrt_error/(h*w))**(1/5)


def main(input_path, k, output_path):
    img = cv2.imread(input_path)
    result = svd(img, k)
    cv2.imwrite(output_path, result)
    
    # checking the images sizes in bytes
    print(f'-----------------RESULTS FOR {input_path} k={k}-----------------')
    p = compression_rate(input_path, output_path)
    print(f'compression rate: {round(p,2)}')
    # calculating the RMSE
    rmse = RMSE(img, result)
    print(f'RMSE: {round(rmse,2)}')


#gets the args from the script call

parser = argparse.ArgumentParser()

# Input
parser.add_argument('-i',
                    '--input',
                    required=False,
                    help='Selected image')

# Input
parser.add_argument('-k',
                    '--k',
                    required=False,
                    help='Numero de Componentes')

# Output
parser.add_argument('-o',
                    '--output',
                    required=False,
                    help='Nome da imagem de saída')

args = parser.parse_args()
input_path = args.input
k = args.k
if k != None:
    k = int(k)
output_path = args.output


#If one arg isnt passed, calculate from all inputs and k={1,5,10,20,30,40,50}
if input_path == None or k == None or output_path == None:
    inputs = ['baboon.png', 'peppers.png', 'monalisa.png','watch.png']
    ks = [1,5,10,20,30,40,50]
    for i in inputs:
        for k in ks:
            main(i, k, ('output/o-'+i[:-4]+'-'+str(k)+'.png'))
else:
    main(input_path, k, output_path)











