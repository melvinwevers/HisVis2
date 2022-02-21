from fastai.vision.all import *
import PIL
import torchvision.transforms as T
import argparse
import glob
import progressbar



def load_model():
    model_ = load_learner('../models/rotation.pkl')
    classes = ['incorrect', 'normal']
    return model_, classes


def label_func(fname):
    return fname.parent.name


def check_orientation(model_, classes, img):
    for rotation in [0, 90, -90, 180]:
        img_ = PIL.Image.open(img).convert('RGB').rotate(rotation, expand=True)
        img_array = np.array(img_)


        label, y, output = model_.predict(img_array)
        probs, preds = output.topk(1)

        top_classes = [classes[i] for i in preds]

        if ((top_classes[0] == 'normal') and (rotation == 0)):
            break
        elif top_classes[0] == 'normal':
            rotated_images[img] = rotation
            img_.save(img)
            break
        elif top_classes[0] == 'incorrect':
            pass

        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='Path to image directory')
    args = parser.parse_args()

    input_path = args.input_path

    rotated_images = {}

    model_, classes = load_model()
    imgs = glob.glob(input_path)
    with progressbar.ProgressBar(max_value = len(imgs)) as bar:
        for i, image in enumerate(imgs):
            imgPath = os.path.join(input_path, image)
            check_orientation(model_, classes, imgPath)
            bar.update(i)

    
    try:
        with open('rotated_images_batch01.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            for key, value in rotated_images.items():
                writer.writerow([key, value])
    except IOError:
        print('I/O errortje!')


