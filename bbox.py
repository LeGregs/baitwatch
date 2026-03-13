import pandas as pd
import numpy as np

# Local
from baitwatch.data import get_labels
from baitwatch.settings import dataset_settings, preprocessing_settings, BUCKET_NAME, DATASET_NAME

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
    img_dataset = [img for img in img_dataset]

    for bb in labels_bb_df:
        img_with_bb = img_dataset.iloc[bb["file_idx"]]

        center_x = bb["center_x"]
        center_y = bb["center_y"]
        width = int(bb["width"])
        height = int(bb["height"])

        cropped_img.append(
            img_with_bb[
                int(center_y - width / 2) : int(center_y + width / 2) + 1,
                int(center_x - height / 2) : int(center_x + height / 2) + 1,
                :
            ]
        )
        class_bb.append(bb["class_id"])

    return cropped_img, class_bb
