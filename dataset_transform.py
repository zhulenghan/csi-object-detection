import os.path as osp

import mmcv

import pandas as pd

from mmengine.fileio import dump, load
from mmengine.utils import track_iter_progress

frame_idx = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90]

def convert_wimans_to_coco(out_file):
    df = pd.read_csv("/home/multisig/datasets/wimans/annotation.csv")

    annotations = []
    images = []
    obj_count = 0
    img_count = 0

    for idx, (row_idx, v) in enumerate(track_iter_progress(list(df.iterrows()))):
        filename = v["label"] + ".npy"
        height, width = (1080, 1920)
        environment = v["environment"]
        wifi_band = v["wifi_band"]

        try:
            df_vid = pd.read_csv("/home/multisig/datasets/wimans/box/" + v["label"] + ".csv", header=None)
        except pd.errors.EmptyDataError:
            continue

        for frame in frame_idx:
            rows = df_vid[df_vid[0] == frame]

            images.append(
                dict(id=img_count, filename=filename, frame=frame,height=height, width=width, environment=environment, wifi_band=wifi_band)
            )

            for _, row in rows.iterrows():

                x_min, y_min, x_max, y_max = row[1:5]

                data_anno = dict(
                    image_id=img_count,
                    id=obj_count,
                    category_id=0,
                    bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                    area=(x_max - x_min) * (y_max - y_min),
                    iscrowd=0)
                annotations.append(data_anno)
                obj_count += 1
            img_count += 1
        
    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{
            'id': 0,
            'name': 'person'
        }]
    )
    dump(coco_format_json, out_file)

if __name__ == '__main__':
    convert_wimans_to_coco("datasets/wimans/annotation_coco_10x.json")

