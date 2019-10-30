import numpy as np
import cv2
import sys


def hide_image(img_input, img2hide, plain):

    #getting the image shapes
    img1_s = img_input.shape
    img2_s = img2hide.shape
    
    #resize seccond image to match the first
    img_resized = cv2.resize(img2hide,(img1_s[0],img1_s[1]))
    #converting the seccond image to 1 bit depth
    r,img2_1b = cv2.threshold(img_resized, 127, 1, cv2.THRESH_BINARY)

    #store how the seccond image shoud be
    cv2.imwrite('img2hide_threshold.png', img2_1b*255)

    # wipe the LSB
    img_input = np.bitwise_and(img_input,254)
    # store the img2hide bit
    img_input = np.bitwise_or(img_input,img2_1b)


    cv2.imwrite('result.png',img_input)

    return img_input


def recover_hidden_image(img):
    result = np.bitwise_and(img,1)
    cv2.imwrite('recovered.png',result*255)


def calc_msg_max_len(img):
    #returns the max quantity of 8bit words that fits in the image
    s = img.shape
    return int( (9/24)*(s[0]*s[1]) )


def get_bits(c):
    r = []
    for i,e in zip(range(7,-1,-1),[1,2,4,8,16,32,64,128][::-1]):
        #print(i,' ',e)
        #print(f'result from applying mask {bin(e)} to  {c}: \t'+
        #       ' {bin(c & e)}')
        #print(f'shifted {i} times: {bin((c&e)>>i)}\n')
        r.append(((ord(c)&e)>>i))
    return r


def hide_string(img, string, bit_plain=0, store_path = 'string_stored.png'):
    
    # calc and print how many chars fits on the image
    max_chars = calc_msg_max_len(img)
    print(f'img able to store {max_chars} chars.'+
            f' String with {len(string)} chars')
    if max_chars < len(string):
        print('the image can\'t store the whole content')

    # adding a end char to the string

    #bit sequence with all bits of the string
    bs = []
    for s in string:
        bs.append(get_bits(s))
    bs = np.array(bs)
    bs.resize(img.shape)

    # creates a mask to wipe the bit_plane but just until the pixel needed to store the characters
    #'''
    mask = np.ones(len(string)*8, dtype='uint8')
    mask.resize(img.shape)
    mask = (mask)*2**bit_plain
    # wipe the right bit of each pixel channel
    img = np.bitwise_and(img,~mask)
    # '''

    # wipe the right bit of each pixel channel without the mask
    #img = np.bitwise_and(img,~(2**bit_plain))
    

    # store the string bits on channels
    img = np.bitwise_or(img, bs<<bit_plain)

    cv2.imwrite(store_path,img)

    return img


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('there is some argument missing')
        print('it needs to be like: python code.py <input.png>'+
              ' <input.txt> <bit_plain> <output.png>')
    else:
        input_path = sys.argv[1]
        input_text_path = sys.argv[2]
        bit_plain = int(sys.argv[3])
        output_path = sys.argv[4]

        img1 = cv2.imread(input_path)
        input_file = open(input_text_path, 'r')
        string = input_file.read()
        stored = hide_string(img1, string, bit_plain=bit_plain, store_path=output_path)
        print('finished coding')
