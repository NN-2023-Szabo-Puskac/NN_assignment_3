import os
import glob
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageFilter
from random import choice, randrange, randint
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait, as_completed
from image_transformer import ImageTransformer, load_image
from datetime import datetime
from time import sleep


NUM_OF_CPU_CORES = 12  # includes logical
BORDER_SIZE = 100
IMG_SHAPE = (222, 310)  # pngs with no background
NO_OF_AUGS = 5
SOURCE_DIR = "converted"
OUTPUT_DIR = "augmented_test"
MIN_ROTATION = 0
MAX_ROTATION = 45
LOG_FILE_NAME = "./data/labels_test.csv"
IMG_COUNTS = [1]
CHUNK_SIZE = 5000
# IMG_COUNTS = prep_image_counts()
CARD_IMAGES = []
BACKGROUNDS = []
LOGS = []


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


def load_backgrounds(start_time: datetime = None) -> list:
    print("LOADING BACKGROUNDS")

    image_paths = glob.glob("./data/wallpapers/*.jpg") + glob.glob(
        "./data/wallpapers/*.png"
    )
    images = []
    for path in image_paths:
        images.append(load_image(path))

    if start_time is not None:
        current = datetime.now()
        print(f"Elapsed: {current-start_time}")
    return images


def select_background(backgrounds) -> str:
    return choice(backgrounds)


def get_random_size() -> int:
    return choice((144, 240, 360))


def save_image(
    aug_idx: int, path: str, image: Image, logs: list, positions: np.array
) -> str:
    path = path.split("/")
    for idx, segment in enumerate(path):
        if segment == f"{SOURCE_DIR}":
            path[idx] = f"{OUTPUT_DIR}"

        if idx == len(path) - 1:
            suffix = segment.split(".")
            suffix[0] = suffix[0] + "_" + str(aug_idx)
            path[idx] = ".".join(suffix)
    new_path = "/".join(path)
    image.save(new_path)

    return f"{new_path}{assemble_positions_string(positions)}\n"


def save_images(list_of_params):
    logs = []
    for params in list_of_params:
        logs.append(save_image(*params))
    return logs


def prepare_foreground(image, size: tuple = None) -> tuple[object, object]:
    it = ImageTransformer(
        image=image, shape=IMG_SHAPE, border_size=BORDER_SIZE, crop=True
    )
    _, _, gamma = get_angles()
    rotated, copied, _ = it.rotate_along_axis(gamma=gamma)
    rotated_img = transforms.ToPILImage()(rotated)
    rotated_copy = transforms.ToPILImage()(copied)
    if size is not None:
        rotated_img = rotated_img.resize((size[0], size[1]))
        rotated_copy = rotated_copy.resize((size[0], size[1]))
    return rotated_img, rotated_copy


def find_position(bg_copy: Image, treshold: int = 1) -> np.array:
    copy = np.array(bg_copy)
    if treshold is not None:
        copy[copy < treshold] = 0
        copy[copy >= treshold] = 255

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


def rotate_background(
    bg_transformer: ImageTransformer, background: Image, copies: list[object]
) -> tuple[object, list]:
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
    jitter = transforms.ColorJitter(
        brightness=(0.9, 1.25), contrast=(0.85, 1.15), saturation=(0.9, 1.25), hue=0.3
    )
    to_jitter = transforms.ToPILImage()(np.array(image)[:, :, :3])
    jittered_image = np.array(jitter(to_jitter))
    image_array = np.array(image)
    image_array[:, :, :3] = jittered_image
    jittered_image = transforms.ToPILImage()(image_array)
    blurred_image = jittered_image.filter(ImageFilter.GaussianBlur())
    return blurred_image


def augmentation_logic(image, backgrounds: list = None) -> tuple[object, np.array]:
    images_count = choice(IMG_COUNTS)

    if backgrounds is None:
        backgrounds = load_backgrounds()

    bg_img = select_background(backgrounds)
    bg = ImageTransformer(image=bg_img, shape=None, border_size=0)
    background = transforms.ToPILImage()(bg.image).convert("RGBA")

    bg_copies = []
    for _ in range(images_count):
        size = (516, 720)  # get_random_size()

        foreground, copy = prepare_foreground(image, size)
        x = randint(0, background.size[0] - foreground.size[0])
        y = randint(0, background.size[1] - foreground.size[1])
        background.paste(foreground, (x, y), foreground)

        background_copy = np.zeros_like(background)
        background_copy = transforms.ToPILImage()(background_copy)
        background_copy.paste(copy, (x, y), copy)
        bg_copies.append(background_copy)

    rotated_bg, rotated_copies = rotate_background(bg, background, bg_copies)
    return jitter_and_blur(rotated_bg), find_positions(rotated_copies)
    # return rotated_bg, find_positions(rotated_copies)


def assemble_positions_string(positions: np.array) -> str:
    positions = positions.flatten()
    size = 20 - len(positions)
    positions = np.pad(positions, (0, size), "constant")
    out = ""
    for position in positions:
        out += f",{position}"
    return out


def augment_image(**kwargs) -> None:
    path_image_tuple = kwargs.get("card")
    logs = kwargs.get("logs")
    backgrounds = kwargs.get("backgrounds")

    outs = []
    for idx in range(NO_OF_AUGS):
        aug_img, positions = augmentation_logic(path_image_tuple[1], backgrounds)
        outs.append((idx + 1, path_image_tuple[0], aug_img, logs, positions))
    return save_images(outs)


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


def parallel_run(logs):
    with ProcessPoolExecutor(max_workers=NUM_OF_CPU_CORES) as executor:
        futures = [
            executor.submit(
                augment_image,
                card=path_image_tuple,
                logs=logs,
                backgrounds=BACKGROUNDS,
            )
            for path_image_tuple in CARD_IMAGES
        ]
        [LOGS.extend(future.result()) for future in as_completed(futures)]


def convert_source_images(start_time):
    if not os.path.exists("./data/converted/"):
        convert_jpg_to_png()
    else:
        print("CONVERTED IMAGES ALREADY EXIST")
        current = datetime.now()
        print(f"Elapsed: {current-start_time}")


def get_paths_to_non_plane_cards(start_time):
    cards = []
    for card in os.listdir(f"./data/{SOURCE_DIR}/"):
        cards.append(f"./data/{SOURCE_DIR}/{card}")

    print("CHECKING CARD SHAPE")
    counter = 0
    non_plane_cards = []
    for card in cards:
        size = Image.open(card).size
        if size[0] >= 400:
            counter += 1
            continue
        non_plane_cards.append(card)
    print(f"DROPPED {counter} CARDS WITH UNSUITABLE SHAPE")
    current = datetime.now()
    print(f"Elapsed: {current-start_time}")
    return non_plane_cards


def load_source_images(paths_to_images, start_time):
    print(f"LOADING MAGIC CARDS DATASET")
    for path in paths_to_images:
        CARD_IMAGES.append((path, load_image(path)))
    current = datetime.now()
    print(f"DATASET LOADED | Elapsed: {current-start_time}")


if __name__ == "__main__":
    start = datetime.now()
    convert_source_images(start_time=start)
    non_plane_cards = get_paths_to_non_plane_cards(start_time=start)
    non_plane_cards = non_plane_cards[:2000]

    # if not os.path.isdir(f"./data/{OUTPUT_DIR}"):
    os.makedirs(f"./data/{OUTPUT_DIR}", exist_ok=True)

    BACKGROUNDS = load_backgrounds(start_time=start)
    load_source_images(non_plane_cards, start)

    print("AUGMENTING CARDS")
    parallel_run(LOGS)

    current = datetime.now()
    print(f"AUGMENTATIONS DONE | Elapsed: {current-start}")

    with open(LOG_FILE_NAME, "w") as data_pairs:
        data_pairs.write(
            "imagename,x1,y1,w1,h1,x2,y2,w2,h2,x3,y3,w3,h3,x4,y4,w4,h4,x5,y5,w5,h5\n"
        )
        for line in LOGS:
            data_pairs.write(line)

    # else:
    #     print("AUGMENTATIONS ALREADY EXIST")
    print("DONE")
