import numpy as np
import cv2
import sys

def convert_back_to_char(binary):
    result = 0
    for i,b in zip(range(7,-1,-1), binary):
        result += (2**i)*b
    return chr(result)


def print_plains(img, path):
        # generate the plain 0
        plain = np.bitwise_and(img,2**0)
        plain *= 255
        cv2.imwrite(output_path[:-4]+'-0.png', plain)
        plain_c = plain.copy()
        plain_c[:,:,1] = 0
        plain_c[:,:,2] = 0
        cv2.imwrite(output_path[:-4]+'-0-b.png', plain_c)
        plain_c = plain.copy()
        plain_c[:,:,0] = 0
        plain_c[:,:,2] = 0
        cv2.imwrite(output_path[:-4]+'-0-g.png', plain_c)
        plain_c = plain.copy()
        plain_c[:,:,0] = 0
        plain_c[:,:,1] = 0
        cv2.imwrite(output_path[:-4]+'-0-r.png', plain_c)
        # generate the plain 1
        plain = np.bitwise_and(img,2**1)>>1
        plain *= 255
        cv2.imwrite(output_path[:-4]+'-1.png', plain)
        plain_c = plain.copy()
        plain_c[:,:,1] = 0
        plain_c[:,:,2] = 0
        cv2.imwrite(output_path[:-4]+'-1-b.png', plain_c)
        plain_c = plain.copy()
        plain_c[:,:,0] = 0
        plain_c[:,:,2] = 0
        cv2.imwrite(output_path[:-4]+'-1-g.png', plain_c)
        plain_c = plain.copy()
        plain_c[:,:,0] = 0
        plain_c[:,:,1] = 0
        cv2.imwrite(output_path[:-4]+'-1-r.png', plain_c)
        # generate the plain 2
        plain = np.bitwise_and(img,2**2)>>2
        plain *= 255
        cv2.imwrite(output_path[:-4]+'-2.png', plain)
        plain_c = plain.copy()
        plain_c[:,:,1] = 0
        plain_c[:,:,2] = 0
        cv2.imwrite(output_path[:-4]+'-2-b.png', plain_c)
        plain_c = plain.copy()
        plain_c[:,:,0] = 0
        plain_c[:,:,2] = 0
        cv2.imwrite(output_path[:-4]+'-2-g.png', plain_c)
        plain_c = plain.copy()
        plain_c[:,:,0] = 0
        plain_c[:,:,1] = 0
        cv2.imwrite(output_path[:-4]+'-2-r.png', plain_c)
        # generate the plain 7
        plain = np.bitwise_and(img,2**7)>>7
        plain *= 255
        cv2.imwrite(output_path[:-4]+'-7.png', plain)
        plain_c = plain.copy()
        plain_c[:,:,1] = 0
        plain_c[:,:,2] = 0
        cv2.imwrite(output_path[:-4]+'-7-b.png', plain_c)
        plain_c = plain.copy()
        plain_c[:,:,0] = 0
        plain_c[:,:,2] = 0
        cv2.imwrite(output_path[:-4]+'-7-g.png', plain_c)
        plain_c = plain.copy()
        plain_c[:,:,0] = 0
        plain_c[:,:,1] = 0
        cv2.imwrite(output_path[:-4]+'-7-r.png', plain_c)


    
def recover_hidden_text(img,bit_plain=0):
    # recover just the LSB from image
    hidden_message = np.bitwise_and(img,2**bit_plain)>>bit_plain

    # reshape into a list of 8bit lists 
    hidden_message = hidden_message.reshape(-1,8)
    #print(f'hidden_message.shape {hidden_message.shape}')
    # concatenate each 8bit element as string to a single string
    string = ''
    for b in hidden_message:
        if (b == [0,0,0,0,0,0,0,0]).all():
            break
        string+=convert_back_to_char(b)

    return string


if __name__ == '__main__':
#python decodificar.py imagem_saida.png plano_bits texto_saida.txt
    if len(sys.argv) < 4:
        print('there is some argument missing')
        print('It needs to be like: python uncode.py <input.png>'+
              ' <bit_plain> <output.txt>')
    else:
        input_path = sys.argv[1]
        bit_plain = int(sys.argv[2])
        output_path = sys.argv[3]

        # read the coded image
        img = cv2.imread(input_path)
        # extract the message and the bitplain
        string = recover_hidden_text(img, bit_plain)
        
        # save the result in a txt file
        output_file = open(output_path, 'w+')
        output_file.write(string)

        # save the plain as image
        print_plains(img,output_path)
        print('finished uncoding')
