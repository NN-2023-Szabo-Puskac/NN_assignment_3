import os
import glob
import more_itertools
import torchvision.transforms as transforms

from PIL import Image
from functools import wraps
from random import choice, randrange, randint
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait
from image_transformer import ImageTransformer

NUM_OF_CPU_CORES = 10  # includes logical
BORDER_SIZE = 100
IMG_SHAPE = None  # pngs with no background
NO_OF_AUGS = 5
SOURCE_DIR = "converted"
OUTPUT_DIR = "augmented_2"
MIN_ROTATION = 0
MAX_ROTATION = 45
LOG_FILE_NAME = "./data/image_label_pairs.csv"

def get_angles():
    x, y, z = choice((-1, 1)), choice((-1, 1)), choice((-1, 1))
    theta, phi, gamma = randrange(MIN_ROTATION, MAX_ROTATION), randrange(MIN_ROTATION, MAX_ROTATION), randrange(MIN_ROTATION, MAX_ROTATION*2)
    return theta*x, phi*y, gamma*z


def select_background():
    images = glob.glob("./data/table_images/*.jpg")
    random_image = choice(images)
    return Image.open(random_image).resize((500, 500)).convert('RGBA')


def get_random_size():
    return choice((144, 240, 360))


def save_image(aug_idx: int, path: str, image: Image):
    path = path.split('/')
    for idx, segment in enumerate(path):
        if segment == f"{SOURCE_DIR}":
            path[idx] = f"{OUTPUT_DIR}"
        
        if idx == len(path) - 1:
            suffix = segment.split('.')
            suffix[0] = suffix[0] + "_" + str(aug_idx+1)
            path[idx] = '.'.join(suffix)
    new_path = '/'.join(path)
    image.save(new_path)
    return new_path


def augmentation_logic(aug_idx, path):
    it = ImageTransformer(path, IMG_SHAPE, BORDER_SIZE)
    theta, phi, gamma = get_angles()
    size = 360  # get_random_size()
    rotated = it.rotate_along_axis(theta=theta, phi=phi, gamma=gamma)
    rotated_img = transforms.ToPILImage()(rotated)
    foreground = rotated_img.resize((size, size))

    background = select_background()
    x = randint(0, background.size[0] - foreground.size[0])
    y = randint(0, background.size[1] - foreground.size[1])
    background.paste(foreground, (x, y), foreground)
    return background, x, y, size
    

def augment_image(path, **kwargs):
    log_name = kwargs.get("log_name")
    log = open(log_name, 'a')

    for idx in range(NO_OF_AUGS):
        aug_img, x, y, size = augmentation_logic(idx, path)
        new_path = save_image(idx, path, aug_img)
        if log:
            log.write(f"{new_path},{x},{y},{size},{size}\n")
    
    log.close()
    print(path)
    

def split(list_, n):
    k, m = divmod(len(list_), n)
    return (list_[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def convert_jpg_to_png():
    os.mkdir("./data/converted/")
    
    list_ = os.listdir('./data/images/')
    for idx, dir in enumerate(list_):
        print(f"Processing image #{idx} of {len(list_)}")
        name = dir.replace('.jpeg','.png')
        img_path = f'./data/images/{dir}'
        new_path = f'./data/converted/{name}'
    
        image = Image.open(img_path)

        # Specifying the RGB mode to the image
        image = image.convert('RGBA')

        # Converting an image from PNG to JPG format
        image.save(new_path)


if __name__ == "__main__":
    if not os.path.exists("./data/converted/"):
        convert_jpg_to_png()
    else:
        print("CONVERTED IMAGES ALREADY EXIST")
  
    if not os.path.isdir(f'./data/{OUTPUT_DIR}'):
        os.mkdir(f'./data/{OUTPUT_DIR}')
        
        cards = []
        for card in os.listdir(f'./data/{SOURCE_DIR}/'):
            cards.append(f'./data/{SOURCE_DIR}/{card}')
        
        data_pairs = open(LOG_FILE_NAME, 'w')
        data_pairs.write("imagename,x,y,w,h\n")
        data_pairs.close()

        with ProcessPoolExecutor(max_workers=NUM_OF_CPU_CORES) as executor:
            futures = [executor.submit(augment_image, image_path, log_name=LOG_FILE_NAME) for image_path in cards[:10]]
            wait(futures)
    else:
        print("AUGMENTATIONS ALREADY EXIST")
    
    print("DONE")
