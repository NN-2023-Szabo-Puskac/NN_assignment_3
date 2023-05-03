import numpy as np
import cv2
from math import pi, floor, sqrt
import torchvision.transforms as transforms

# Source: https://github.com/eborboihuc/rotate_3d + some minor changes to make it for for this use case
# Parameters:
#     image_path: the path of image that you want rotated
#     shape     : the ideal shape of input image, None for original size.
#     theta     : rotation around the x axis
#     phi       : rotation around the y axis
#     gamma     : rotation around the z axis (basically a 2D rotation)
#     dx        : translation along the x axis
#     dy        : translation along the y axis
#     dz        : translation along the z axis (distance to the image)
#
# Output:
#     image     : the rotated image
#
# Reference:
#     1.        : http://stackoverflow.com/questions/17087446/how-to-calculate-perspective-transform-for-opencv-from-rotation-angles
#     2.        : http://jepsonsblog.blogspot.tw/2012/11/rotation-in-3d-using-opencvs.html

""" Utility Functions """


def load_image(img_path, shape=None):
    if shape is not None:
        img = cv2.imread(img_path)
        img = cv2.resize(img, shape)
    else:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    return img


def save_image(img_path, img):
    cv2.imwrite(img_path, img)


def get_rad(theta, phi, gamma):
    return (deg_to_rad(theta), deg_to_rad(phi), deg_to_rad(gamma))


def get_deg(rtheta, rphi, rgamma):
    return (rad_to_deg(rtheta), rad_to_deg(rphi), rad_to_deg(rgamma))


def deg_to_rad(deg):
    return deg * pi / 180.0


def rad_to_deg(rad):
    return rad * 180.0 / pi


def calculate_diameter(size):
    return floor(sqrt(size[0] ** 2 + size[1] ** 2) / 2) - 10


class ImageTransformer(object):
    """Perspective transformation class for image
    with shape (height, width, #channels)"""

    def __init__(
        self,
        image_path: str = None,
        shape: tuple = None,
        border_size: int = 0,
        crop: bool = False,
        image=None,
    ):
        self.image_path = image_path

        if image is None:
            self.image = load_image(image_path, None)
        else:
            self.image = image

        if crop:
            self.resize_and_crop(shape)

        self.copy = 255 * np.ones_like(self.image)
        self.copy = self.prepare_borders(self.copy, border_size)
        self.image = self.prepare_borders(self.image, border_size)

    def prepare_borders(self, image, border_size):
        new_img = cv2.copyMakeBorder(
            image,
            top=border_size,
            bottom=border_size,
            left=border_size,
            right=border_size,
            borderType=cv2.BORDER_CONSTANT,
        )
        self.height = new_img.shape[0]
        self.width = new_img.shape[1]
        self.num_channels = new_img.shape[2]
        return new_img

    def resize(self, size):
        self.image = np.array(transforms.ToPILImage()(self.image).resize(size))

    def resize_and_crop(self, size: tuple[int, int]):
        img = np.array(transforms.ToPILImage()(self.image).resize(size))
        dia = calculate_diameter(size)

        mask = np.zeros_like(img)
        cv2.circle(
            mask,
            (floor(size[0] / 2), floor(size[1] / 2)),
            dia,
            (255, 255, 255),
            thickness=-1,
        )
        mask = mask.astype(np.uint8)

        masked = img * (mask / 255)
        masked[:, :, 3] = img[:, :, 3]

        self.mask = mask
        self.image = masked.astype(np.uint8)

    """ Wrapper of Rotating a Image """

    def rotate_along_axis(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0):
        # Get radius of rotation along 3 axes
        rtheta, rphi, rgamma = get_rad(theta, phi, gamma)

        # Get ideal focal length on z axis
        # NOTE: Change this section to other axis if needed
        d = np.sqrt(self.height**2 + self.width**2)
        self.focal = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)
        dz = self.focal

        # Get projection matrix
        mat = self.get_M(rtheta, rphi, rgamma, dx, dy, dz)

        image = cv2.warpPerspective(self.image.copy(), mat, (self.width, self.height))
        copy = cv2.warpPerspective(self.copy.copy(), mat, (self.width, self.height))
        return image, copy, mat

    """ Get Perspective Projection Matrix """

    def get_M(self, theta, phi, gamma, dx, dy, dz):
        w = self.width
        h = self.height
        f = self.focal

        # Projection 2D -> 3D matrix
        A1 = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1], [0, 0, 1]])

        # Rotation matrices around the X, Y, and Z axis
        RX = np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(theta), -np.sin(theta), 0],
                [0, np.sin(theta), np.cos(theta), 0],
                [0, 0, 0, 1],
            ]
        )
        RY = np.array(
            [
                [np.cos(phi), 0, -np.sin(phi), 0],
                [0, 1, 0, 0],
                [np.sin(phi), 0, np.cos(phi), 0],
                [0, 0, 0, 1],
            ]
        )
        RZ = np.array(
            [
                [np.cos(gamma), -np.sin(gamma), 0, 0],
                [np.sin(gamma), np.cos(gamma), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        # Composed rotation matrix with (RX, RY, RZ)
        R = np.dot(np.dot(RX, RY), RZ)

        # Translation matrix
        T = np.array([[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz], [0, 0, 0, 1]])

        # Projection 3D -> 2D matrix
        A2 = np.array([[f, 0, w / 2, 0], [0, f, h / 2, 0], [0, 0, 1, 0]])

        # Final transformation matrix
        return np.dot(A2, np.dot(T, np.dot(R, A1)))
