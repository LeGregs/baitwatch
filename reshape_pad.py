import tensorflow as tf
import cv2 as cv
import pandas as pd


def reshape_pad_crop(cropped_img, format_img = (105,256)):
    bb_crop_fin = []
    for img in cropped_img:
        img_proc = img.astype("uint8")

        ratio = max(img_proc.shape[0]/format_img[0], img_proc.shape[1]/format_img[1])

        img_resize = cv.resize(img_proc, (int(img_proc.shape[1]/ratio),
                                    int(img_proc.shape[0]/ratio)))

        bb_crop_fin.append(tf.image.pad_to_bounding_box(img_resize, format_img[0] - img_resize.shape[0],
                                format_img[1] - img_resize.shape[1],
                                format_img[0],
                                format_img[1]))
    return bb_crop_fin
