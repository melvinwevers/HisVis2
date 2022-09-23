#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import cv2 as cv
import os
import glob
from PIL import Image
import time


def crop_biggest(imgPath, outPath, format, threshold = 50, resize=True, find_border=True):
    w = 0
    h = 0
    # set boundaries for cropped region, dependent on picture type
    if format == '6x6':
        w_min = 1250
        h_min = 1250
        h_max = 1300
        w_max = 1300
    else:
        w_min = 750
        w_max = 900
        h_min = 575
        h_max = 625
    try: 
        img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
        file_name = os.path.basename(imgPath)[:-4]
        #print(file_name)
        dir_name = os.path.dirname(glob.glob(imgPath)[0]).split('/')[-1]
    except Exception:
        print('File could not be opened')
        pass


    if resize:
        scale_percent = 25

        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        
        # dsize
        resized = (width, height)

        # resize image
        img = cv.resize(img, resized)

    if find_border:
        while w < w_min or h < h_min:
            _, threshold_ = cv.threshold(img, threshold, 255,0)
            contours, _ = cv.findContours(threshold_, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            contour_sizes = [(cv.contourArea(contour), contour) for contour in contours]

            biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
            x, y, w, h = cv.boundingRect(biggest_contour)

            threshold -= 1 # lower threshold if images is too small
            if threshold < 45:
                #print('stop lowering threshold')
                mistakes.append(file_name)
                break
        if w > w_max or h > h_max:
            # if images are still too large add their filenames to exported list
            mistakes.append(file_name)
            pass
        
        roi = img[y  :y + h, x : x + w ]
    else:
        roi = img
    export_file_name = ("{}.jpg".format(file_name))
    if not os.path.exists(os.path.join(outPath, dir_name)):
        os.makedirs(os.path.join(outPath, dir_name))
    cv.imwrite(os.path.join(outPath, dir_name, export_file_name), roi)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='Path to image directory') # '../data/**/*.jp2' Don't forget the ' '
    parser.add_argument('-o', '--output_path', help='Output Path')
    parser.add_argument('-t', '--threshold', help='border detection threshold', default=50)
    parser.add_argument('-f', '--format', help='type of photograph')
    parser.add_argument('-r', '--resize', help='resize images', action='store_true', default=False) #todo add scale
    parser.add_argument('-c', '--crop', help='Crop images', action='store_true', default=False)
    args = parser.parse_args()

    current_time = time.strftime("%Y%m%d")

    input_path = args.input_path
    format = args.format
    output_path = args.output_path 
    threshold = args.threshold
    resize = args.resize
    find_border = args.crop
    print("crop:", find_border)
    print("resize:", resize)
    print("format:", format)
    mistakes = []

    if not os.path.exists(output_path):
        os.makedirs(output_path)


    imgs = glob.glob(input_path, recursive=True)
    print(len(imgs))

    for i, image in enumerate(imgs):
        if i % 100 == 99:
            print(i)
        #imgPath = os.path.join(input_path, image)
        crop_biggest(image, output_path, format, threshold, resize, find_border)
        os.remove(image)


    with open(os.path.join(output_path, f'{current_time}_mistakes_cropping.txt'), 'w') as f:
        for item in mistakes:
            f.write("%s\n" % item)
        

    
    


    
