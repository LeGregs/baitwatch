import pandas as pd
import numpy as np
import tensorflow as tf
import cv2 as cv
import pandas as pd
from tensorflow.data import AUTOTUNE

# Local
from baitwatch.data import get_labels, get_images
from baitwatch.settings import dataset_settings, preprocessing_settings, BUCKET_NAME, DATASET_NAME
from baitwatch.preprocessing import preprocess
from baitwatch.data import save_image_dataset


def build_bbox_dataframe(labels_dataset,IMG_SIZE=(1920,1080)):
    IMG_W, IMG_H = IMG_SIZE
    rows = []
    for idx, txt in enumerate(labels_dataset.as_numpy_iterator()):
        txt = txt.decode("utf-8").strip()
        if txt == "":
            continue
        for line in txt.split("\n"):
            parts = line.split(" ")
            class_id = int(parts[0])
            center_x = float(parts[1]) * IMG_W
            center_y = float(parts[2])* IMG_H
            w = float(parts[3])* IMG_W
            h = float(parts[4])* IMG_H
            rows.append({
                "file_idx": idx,
                "class_id": class_id,
                "center_x": center_x,
                "center_y": center_y,
                "width": w,
                "height": h,
                "area": w * h
            })
    return pd.DataFrame(rows)



def crop_bb(labels_bb_df, img_dataset):
    cropped_img = []
    class_bb = []
    img_df = []
    for ten in img_dataset:
        img_df.append(ten.numpy())

    for bb in range(len(labels_bb_df)):
        num_img = labels_bb_df.iloc[bb]['file_idx']
        img_with_bb = img_df[int(num_img)]
        bb_label = labels_bb_df.iloc[bb]
        center_x = bb_label.loc["center_x"]
        center_y = bb_label.loc["center_y"]
        width = bb_label.loc["width"]
        height = bb_label.loc["height"]

        cropped_img.append(img_with_bb[
            int(center_y - height/2) : int(center_y + height/2) + 1,
            int(center_x - width/2) : int(center_x + width/2) + 1,:])
        class_bb.append(int(labels_bb_df.iloc[bb]["class_id"]))

    return cropped_img, class_bb




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


def get_dataset_IFSP():

    imgs_train, imgs_val, imgs_test = get_images()
    imgs_train_preprocessed = imgs_train.map(preprocess, num_parallel_calls=AUTOTUNE)
    imgs_val_preprocessed = imgs_val.map(preprocess, num_parallel_calls=AUTOTUNE)
    imgs_test_preprocessed = imgs_test.map(preprocess, num_parallel_calls=AUTOTUNE)

    lab_train, lab_val, lab_test = get_labels()

    bb_df_train = build_bbox_dataframe(lab_train)
    bb_df_val = build_bbox_dataframe(lab_val)
    bb_df_test = build_bbox_dataframe(lab_test)

    crop_train, y_train_ifsp = crop_bb(bb_df_train, imgs_train_preprocessed)
    crop_val, y_val_ifsp = crop_bb(bb_df_val, imgs_val_preprocessed)
    crop_test, y_test_ifsp = crop_bb(bb_df_test, imgs_test_preprocessed)

    X_train_ifsp = tf.data.Dataset.from_tensor_slices(reshape_pad_crop(crop_train))
    X_val_ifsp = tf.data.Dataset.from_tensor_slices(reshape_pad_crop(crop_val))
    X_test_ifsp = tf.data.Dataset.from_tensor_slices(reshape_pad_crop(crop_test))

    save_image_dataset(X_train_ifsp,
                       dataset_settings.PROCESSED_DATA_PATH / "ifsp" / "train",
                       labels=np.array(y_train_ifsp))
    save_image_dataset(X_val_ifsp,
                       dataset_settings.PROCESSED_DATA_PATH / "ifsp" / "val",
                       labels=np.array(y_val_ifsp))
    save_image_dataset(X_test_ifsp,
                       dataset_settings.PROCESSED_DATA_PATH / "ifsp" / "test",
                       labels=np.array(y_test_ifsp))
