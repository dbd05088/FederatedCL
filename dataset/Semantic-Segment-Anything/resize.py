from PIL import Image
import argparse
import os

def resize_img(image_root_dir):
    image_lists = os.listdir(image_root_dir)
    for image_name in image_lists:
        image = Image.open(os.path.join(image_root_dir, image_name))
        w, h = image.size
        resized_image = image.resize((1024, 1024))
        resized_image_name = os.path.join(image_root_dir + "_1024", image_name)
        resized_image.save(resized_image_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get mean and std for images')
    parser.add_argument('--image_root_dir', type=str, help='Root directory of images')
    args = parser.parse_args()
    os.makedirs(args.image_root_dir + "_1024", exist_ok=True)
    resize_img(args.image_root_dir)