from mmdet.datasets import build_dataset
import cv2

dataset_cfg = 'repos/mmdet_wifi/configs/train.py'
dataset = build_dataset(dataset_cfg)

for i in range(10):  # Visualize 10 samples
    data = dataset[i]
    img = cv2.imread(data['img_path'])
    for instance in data['instances']:
        bbox = instance['bbox']
        label = instance['bbox_label']
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        cv2.putText(img, str(label), (int(bbox[0]), int(bbox[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow('Annotation', img)
    cv2.waitKey(0)