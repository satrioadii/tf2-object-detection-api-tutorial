import cv2
import numpy as np
from PIL import Image
import time

def make_image_square(filename):
	img = cv2.imread(filename)
	#Size of the image
	# s = max(img.shape[0:2])
	s = 640

	#Creating a dark square with NUMPY  
	f = np.zeros((s,s,3),np.uint8)

	#Getting the centering position
	ax,ay = (s - img.shape[1])//2,(s - img.shape[0])//2
	
	#Pasting the 'image' in a centering position
	f[ay:img.shape[0]+ay,ax:ax+img.shape[1]] = img
	cv2.imwrite(filename,f)


def crop_image():
    for image_index in range(2):
        folder_name = 'data/'
	    image_name = folder_name + str(image_index + 1) + '.jpg'
        img = cv2.imread(image_name)

        h, w, c = img.shape

        print(h,w, image_name)

        w_constant  = w/3
        h_constant = h/2

        image_part_index = 0

        for index_w in range(3):
            for index_h in range(2):
                start_width = int(w_constant * index_w)
                end_width = int(w_constant * (index_w + 1))

                start_height = int(h_constant * index_h)
                end_height = int(h_constant * (index_h + 1))

                current_index = image_part_index

                # For training image set
                # section_name = 'PL_8_' + str(image_index+1) + '_'
                # file_name = section_name + str(image_index+1) + '_' + str(image_part_index) + '.jpg'

                # For testing image set
                section_name = str(image_index+1) + '/'
                file_name = folder_name + section_name + str(image_part_index+1) + '.jpg'

                crop_img = img[start_height:end_height, start_width:end_width]

                image_part_index = image_part_index + 1
                cv2.imwrite(file_name, crop_img)

                make_image_square(file_name)



def main():

    ratarata=0
    #=========== Put your Program in here ========#
    crop_image()
    time.sleep(5)
    #====== Put the return value to ratarata =====#
    ratarata=200
    #=============================================#

    print(ratarata)

if __name__ == "__main__":
    main()


