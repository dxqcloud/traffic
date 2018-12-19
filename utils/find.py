import cv2
import numpy as np
import os
import time

def find_car(image_list, threshold=10):
    image_num = len(image_list)
    mask = np.zeros((image_num,image_num))

    for i in range(image_num):
        for j in range(i+1, image_num):
            mask[i,j] = mask[j,i] = np.sum((np.abs(image_list[i] - image_list[j])) < threshold)
    mask = np.sum(mask,axis=1)
    return np.argmin(mask)

if __name__ == "__main__":
    image_dir = '/Users/sherry/Desktop/0_1208_res'
    images = os.listdir(image_dir)
    for image in images:
        print(image)
        image = cv2.imread(os.path.join(image_dir, image))
        image_shape = image.shape
        print(image_shape)
        l, w = image_shape[0], image_shape[1]
        image_1, image_2, image_3, image_4 = image[0:l//2, 0:w//2,:],image[l-l//2:l,0:w//2,:],image[0:l//2, w-w//2:w,:],image[l-l//2:l, w-w//2:w,:]

        start_time = time.time()
        print(find_car([image_1, image_2, image_3, image_4]))
        print("Time cost:")
        print(time.time() - start_time)
