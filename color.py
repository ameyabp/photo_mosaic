from PIL import Image
import cv2 as cv
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--image_path', type=str, required=True, help="Path to the input image")

    args = parser.parse_args()
    image_path = args.image_path
    img = cv.imread(image_path, cv.IMREAD_UNCHANGED)

    summary_image = np.zeros(img.shape, np.uint8)
    summary_hsv_image = np.zeros(img.shape, np.uint8)

    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # chunking the image in SxS blocks and recreating the image with average color of each chunk
    # block size in either dimension should not be more than the image width or height
    block_size_width = 10
    block_size_height = 10
    for i in range(0, img.shape[0], block_size_height):
        for j in range(0, img.shape[1], block_size_width):
            # JUST PERFORM CONVOLUTION!!!
            summary_image[i:i+block_size_height, j:j+block_size_width] = \
                (np.mean(img[i:i+block_size_height, j:j+block_size_width, 0]), 
                 np.mean(img[i:i+block_size_height, j:j+block_size_width, 1]), 
                 np.mean(img[i:i+block_size_height, j:j+block_size_width, 2]))
            
            summary_hsv_image[i:i+block_size_height, j:j+block_size_width] = \
                (np.mean(hsv_image[i:i+block_size_height, j:j+block_size_width, 0]), 
                 np.mean(hsv_image[i:i+block_size_height, j:j+block_size_width, 1]), 
                 np.mean(hsv_image[i:i+block_size_height, j:j+block_size_width, 2]))                
    
    cv.imwrite(f'fidelity_{round(block_size_width/img.shape[1],2)}_{round(block_size_height/img.shape[0],2)}_RGB.jpg', summary_image)
    cv.imwrite(f'fidelity_{round(block_size_width/img.shape[1],2)}_{round(block_size_height/img.shape[0],2)}_HSV.jpg', cv.cvtColor(summary_hsv_image, cv.COLOR_HSV2BGR))