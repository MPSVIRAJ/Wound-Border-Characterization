import os
import cv2
import numpy as np

def data_loader(ImageID:str, DATA_PATH:str = '../data') :
    
    IMAGE_PATH = f'{DATA_PATH}/images/{ImageID}.png'
    WOUND_MASK_PATH = f'{DATA_PATH}/wound_masks/{ImageID}.png'
    BODY_MASK_PATH = f'{DATA_PATH}/body_masks/{ImageID}.png'
    MARKER_MASK_PATH = f'{DATA_PATH}/marker_masks/{ImageID}.png'
    DEPTH_MAP_PATH = f'{DATA_PATH}/depth_maps/{ImageID}.png'

    image = cv2.imread(IMAGE_PATH)[..., ::-1]
    wound = cv2.imread(WOUND_MASK_PATH, cv2.IMREAD_GRAYSCALE)
    body = cv2.imread(BODY_MASK_PATH, cv2.IMREAD_GRAYSCALE)
    depth= cv2.imread(DEPTH_MAP_PATH, cv2.IMREAD_ANYDEPTH)
    marker = cv2.imread(MARKER_MASK_PATH, cv2.IMREAD_GRAYSCALE)

    depth = cv2.bitwise_and(depth, depth, mask=body)

    if not marker is None:
        depth = cv2.bitwise_and(depth, depth, mask=~marker)
        body = cv2.bitwise_and(body, body, mask=~marker)
        
    # Put all the loaded data into a dictionary
    loaded_data = {
        'image': image,
        'wound': wound,
        'body': body,
        'depth': depth
    }
    return loaded_data





