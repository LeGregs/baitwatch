"""
Baitwatch — Bounding Box Pipeline
build_bbox_dataframe : parse les labels YOLO → DataFrame
crop_bb : crop les bounding boxes depuis les images
reshape_pad_crop : resize + pad les crops au format cible
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import cv2 as cv
import pandas as pd

# Local
from baitwatch.data import get_labels
from baitwatch.settings import dataset_settings, preprocessing_settings, BUCKET_NAME, DATASET_NAME


def build_bbox_dataframe(labels_dataset, IMG_SIZE=(1920, 1080)):
    """
    Lit les fichiers labels YOLO et renvoie un DataFrame
    avec les coordonnées en pixels de chaque bounding box.
    """
    IMG_W, IMG_H = IMG_SIZE
    rows = []

    print("📄 Lecture des labels YOLO...")

    for idx, txt in enumerate(labels_dataset.as_numpy_iterator()):
        txt = txt.decode("utf-8").strip()
        if txt == "":
            continue
        for line in txt.split("\n"):
            parts = line.split(" ")
            class_id = int(parts[0])
            center_x = float(parts[1]) * IMG_W
            center_y = float(parts[2]) * IMG_H
            w = float(parts[3]) * IMG_W
            h = float(parts[4]) * IMG_H
            rows.append({
                "file_idx": idx,
                "class_id": class_id,
                "center_x": center_x,
                "center_y": center_y,
                "width": w,
                "height": h,
                "area": w * h
            })

    print(f"✅ {len(rows)} bounding boxes extraites depuis {idx + 1} fichiers labels")

    return pd.DataFrame(rows)


def crop_bb(labels_bb_df, img_dataset):
    """
    Crop chaque bounding box depuis les images.
    Renvoie la liste des crops (np.array) et leurs class_id.
    """
    cropped_img = []
    class_bb = []
    img_df = []

    print("✂️  Chargement des images en mémoire...")

    for ten in img_dataset:
        img_df.append(ten.numpy())

    print(f"   {len(img_df)} images chargées")
    print("🔲 Découpe des bounding boxes...")

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
        class_bb.append(labels_bb_df["class_id"])

    print(f"✅ {len(cropped_img)} crops générés")

    return cropped_img, class_bb


def reshape_pad_crop(cropped_img, format_img=(105, 256)):
    """
    Resize chaque crop en gardant le ratio,
    puis pad pour atteindre le format cible (h, w).
    """
    bb_crop_fin = []

    print(f"📐 Resize + pad des crops vers {format_img}...")

    for img in cropped_img:
        img_proc = img.astype("uint8")

        ratio = max(img_proc.shape[0]/format_img[0], img_proc.shape[1]/format_img[1])

        img_resize = cv.resize(img_proc, (int(img_proc.shape[1]/ratio),
                                    int(img_proc.shape[0]/ratio)))

        bb_crop_fin.append(tf.image.pad_to_bounding_box(img_resize, format_img[0] - img_resize.shape[0],
                                format_img[1] - img_resize.shape[1],
                                format_img[0],
                                format_img[1]))

    print(f"✅ {len(bb_crop_fin)} crops reformatés en {format_img}")

    return bb_crop_fin
