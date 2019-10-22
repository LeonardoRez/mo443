import numpy as np
import cv2
import sys

def convert_back_to_char(binary):
    result = 0
    for i,b in zip(range(7,-1,-1), binary):
        result += (2**i)*b
    return chr(result)

    
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

        img = cv2.imread(input_path)
        string = recover_hidden_text(img, bit_plain)
        output_file = open(output_path, 'w+')
        output_file.write(string)
        print('finished uncoding')
