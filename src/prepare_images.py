 
import argparse
import os 
import glob
import progressbar
from remove_borders import crop_biggest
 
def main():
    '''
    Here we remove borders and resize images
    Then we try and find the correct rotation of images and rotate them accordingly
    
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='Path to image directory')
    parser.add_argument('-o', '--output_path', help='Output Path')
    parser.add_argument('-t', '--threshold', help='border detection threshold')
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path if args.output_path else args.input_path
    threshold = args.threshold

    mistakes = []

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    #/Users/melvinwevers/Downloads/Batch_01/**/*.jpg'
    imgs = glob.glob(input_path)
    print(len(imgs))
    with progressbar.ProgressBar(max_value = len(imgs)) as bar:
        for i, image in enumerate(imgs):
            imgPath = os.path.join(input_path, image)
            crop_biggest(imgPath, output_path)
            bar.update(i)

    with open('mistakes.txt', 'w') as f:
        for item in mistakes:
            f.write("%s\n" % item)