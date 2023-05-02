import os
import glob
import more_itertools
import cv2
import sys
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageFilter
from functools import wraps
from random import choice, randrange, randint
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait
from image_transformer import ImageTransformer
from typing import List
from datetime import datetime

NUM_OF_CPU_CORES = 10  # includes logical
BORDER_SIZE = 100
IMG_SHAPE = (222, 310)  # pngs with no background
NO_OF_AUGS = 4
SOURCE_DIR = "converted"
OUTPUT_DIR = "augmented_1"
MIN_ROTATION = 0
MAX_ROTATION = 45
LOG_FILE_NAME = "./data/labels.csv"
IMG_COUNTS = [1]
# IMG_COUNTS = prep_image_counts()


def prep_image_counts() -> list[int]:
    counts = []
    for _ in range(50):
        counts.append(1)
    for _ in range(25):
        counts.append(2)
    for _ in range(15):
        counts.append(3)
    for _ in range(9):
        counts.append(4)
    for _ in range(1):
        counts.append(5)
    return counts


def get_angles(
    max_theta: int = MAX_ROTATION,
    max_phi: int = MAX_ROTATION,
    max_gamma: int = MAX_ROTATION * 2,
) -> tuple[float, float, float]:
    x, y, z = choice((-1, 1)), choice((-1, 1)), choice((-1, 1))
    theta, phi, gamma = (
        randrange(MIN_ROTATION, max_theta),
        randrange(MIN_ROTATION, max_phi),
        randrange(MIN_ROTATION, max_gamma),
    )
    return (theta * x, phi * y, gamma * z)


def select_background() -> str:
    images = glob.glob("./data/wallpapers/*.jpg") + glob.glob("./data/wallpapers/*.png")
    return choice(images)


def get_random_size() -> int:
    return choice((144, 240, 360))


def save_image(aug_idx: int, path: str, image: Image) -> str:
    path = path.split("/")
    for idx, segment in enumerate(path):
        if segment == f"{SOURCE_DIR}":
            path[idx] = f"{OUTPUT_DIR}"

        if idx == len(path) - 1:
            suffix = segment.split(".")
            suffix[0] = suffix[0] + "_" + str(aug_idx + 1)
            path[idx] = ".".join(suffix)
    new_path = "/".join(path)
    image.save(new_path)
    return new_path


def prepare_foreground(path: str, size: int) -> tuple[object, object]:
    it = ImageTransformer(path, IMG_SHAPE, BORDER_SIZE, crop=True)
    _, _, gamma = get_angles()
    rotated, copied, _ = it.rotate_along_axis(gamma=gamma)
    rotated_img = transforms.ToPILImage()(rotated)
    rotated_copy = transforms.ToPILImage()(copied)
    return rotated_img.resize((size, size)), rotated_copy.resize((size, size))


def find_position(bg_copy: Image) -> np.array:
    copy = np.array(bg_copy)
    offsetx = None
    for idx, row in enumerate(copy.transpose(1, 0, 2)):
        if 255 in row:
            offsetx = idx
            break

    width = None
    for idx, row in enumerate(copy.transpose(1, 0, 2)[offsetx:]):
        if 255 not in row:
            width = idx
            break

    offsety = None
    for idx, row in enumerate(copy):
        if 255 in row:
            offsety = idx
            break

    height = None
    for idx, row in enumerate(copy[offsety:]):
        if 255 not in row:
            height = idx
            break

    return np.array([offsetx, offsety, width, height])


def find_positions(bg_copies: list[object]) -> np.array:
    positions = []
    for copy in bg_copies:
        positions.append(find_position(copy))
    return np.array(positions)


def rotate_background(bg_transformer: ImageTransformer, background: Image, copies: list[object]) -> tuple[object, list]:
    theta, phi, gamma = get_angles(max_theta=30, max_phi=30, max_gamma=15)
    bg_transformer.image = bg_transformer.prepare_borders(
        np.array(background), BORDER_SIZE
    )
    rotated_bg, _, _ = bg_transformer.rotate_along_axis(
        theta=theta, phi=phi, gamma=gamma
    )

    rotated_copies = []
    for copy in copies:
        bg_transformer.image = bg_transformer.prepare_borders(
            np.array(copy), BORDER_SIZE
        )
        rotated_bgc, _, _ = bg_transformer.rotate_along_axis(
            theta=theta, phi=phi, gamma=gamma
        )
        rotated_copies.append(transforms.ToPILImage()(rotated_bgc).resize((640, 640)))

    return transforms.ToPILImage()(rotated_bg).resize((640, 640)), rotated_copies


def jitter_and_blur(image: Image) -> Image:
    jitter = transforms.ColorJitter(brightness=0.1, contrast=1, saturation=0.1, hue=0.5)
    to_jitter = transforms.ToPILImage()(np.array(image)[:,:,:3])
    jittered_image = np.array(jitter(to_jitter))
    image_array = np.array(image)
    image_array[:,:,:3] = jittered_image
    jittered_image = transforms.ToPILImage()(image_array)
    blurred_image = jittered_image.filter(ImageFilter.GaussianBlur())
    return blurred_image
    

def augmentation_logic(path: str) -> tuple[object, np.array]:
    images_count = choice(IMG_COUNTS)

    bg_path = select_background()
    bg = ImageTransformer(bg_path, None, 0)
    background = transforms.ToPILImage()(bg.image).convert("RGBA")

    bg_copies = []
    for _ in range(images_count):
        size = 720  # get_random_size()

        foreground, copy = prepare_foreground(path, size)
        x = randint(0, background.size[0] - foreground.size[0])
        y = randint(0, background.size[1] - foreground.size[1])
        background.paste(foreground, (x, y), foreground)

        background_copy = np.zeros_like(background)
        background_copy = transforms.ToPILImage()(background_copy)
        background_copy.paste(copy, (x, y), copy)
        bg_copies.append(background_copy)

    rotated_bg, rotated_copies = rotate_background(bg, background, bg_copies)
    return jitter_and_blur(rotated_bg), find_positions(rotated_copies)


def assemble_positions_string(positions: np.array) -> str:
    positions = positions.flatten()
    size = 20 - len(positions)
    positions = np.pad(positions, (0, size), "constant")
    out = ""
    for position in positions:
        out += f",{position}"
    return out


def augment_image(path: str, **kwargs) -> None:
    log_name = kwargs.get("log_name")
    log = open(log_name, "a")

    for idx in range(NO_OF_AUGS):
        aug_img, positions = augmentation_logic(path)
        new_path = save_image(idx+1, path, aug_img)
        if log:
            log.write(f"{new_path}{assemble_positions_string(positions)}\n")

    log.close()


def convert_jpg_to_png():
    os.mkdir("./data/converted/")

    list_ = os.listdir("./data/images/")
    for idx, dir in enumerate(list_):
        print(f"Processing image #{idx} of {len(list_)}")
        name = dir.replace(".jpeg", ".png")
        img_path = f"./data/images/{dir}"
        new_path = f"./data/converted/{name}"

        image = Image.open(img_path)

        # Specifying the RGB mode to the image
        image = image.convert("RGBA")

        # Converting an image from PNG to JPG format
        image.save(new_path)


if __name__ == "__main__":
    if not os.path.exists("./data/converted/"):
        convert_jpg_to_png()
    else:
        print("CONVERTED IMAGES ALREADY EXIST")

    cards = []
    for card in os.listdir(f"./data/{SOURCE_DIR}/"):
        cards.append(f"./data/{SOURCE_DIR}/{card}")

    print("CHECKING CARD SHAPE")
    counter = 0
    non_plain_cards = []
    for card in cards:
        size = Image.open(card).size
        if size[0] >= 400:
            counter += 1
            continue
        non_plain_cards.append(card)
    print(f"DROPPED {counter} CARDS WITH UNSUITABLE SHAPE")

    if not os.path.isdir(f"./data/{OUTPUT_DIR}"):
        os.mkdir(f"./data/{OUTPUT_DIR}")
        data_pairs = open(LOG_FILE_NAME, "w")
        data_pairs.write(
            "imagename,x1,y1,w1,h1,x2,y2,w2,h2,x3,y3,w3,h3,x4,y4,w4,h4,x5,y5,w5,h5\n"
        )
        data_pairs.close()
    index = 0
    start = datetime.now()
    with ProcessPoolExecutor(max_workers=NUM_OF_CPU_CORES) as executor:
        futures = [
            executor.submit(augment_image, image_path, log_name=LOG_FILE_NAME)
            for image_path in non_plain_cards
        ]
        for future in futures:
            future.result()
            index += 1
            if index % 100:
                current = datetime.now()
                print(
                    f"PROGRESS {index}/{len(non_plain_cards)} | Elapsed: {current-start}"
                )
        # for card in non_plain_cards[:images_to_process]:
        #     augment_image(card, log_name=LOG_FILE_NAME)
    # lse:
    #    print("AUGMENTATIONS ALREADY EXIST")    
    print("DONE")
