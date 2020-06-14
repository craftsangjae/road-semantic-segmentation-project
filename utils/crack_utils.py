"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
import json
import numpy as np
import cv2
import os
from tqdm import tqdm
from .pascal_utils import label_to_color_image
import pandas as pd


# Set-Up Directory
ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__)))
DATASET_DIR = os.path.join(ROOT_DIR, 'datasets')
CRACK_DIR = os.path.join(DATASET_DIR, 'pavement_crack')
SEG_DIR = os.path.join(CRACK_DIR, 'segmentations')
ABSEG_DIR = os.path.join(CRACK_DIR, "abnormal_segmentations")
IMAGE_DIR = os.path.join(CRACK_DIR, "images")
CRACK_ZIP_PATH = os.path.join(CRACK_DIR, "hyundai_crack.zip")
CRACK_JSON_PATH = os.path.join(CRACK_DIR, "crack.json")


LABEL_MAP = {
    "background": 0,
    "other_road": 1,
    "my_road": 2,
    "manhole": 3,
    "car": 4,
    "steel": 5,
    "pothole": 6,
    "bump": 7,
    "abnormal": 8,
    'ignore': 255
}


class CrackDataset:
    """
    Crack Image을 Loading하는 데이터셋
    """

    def __init__(self, data_dir="pavement_crack/", fnames=None):
        if not os.path.exists(data_dir):
            raise ValueError("{}가 존재하지 않습니다.".format(data_dir))
        self.image_dir = os.path.join(data_dir, "images/")
        self.seg_dir = os.path.join(data_dir, "segmentations/")
        if not fnames:
            self.fnames = [os.path.splitext(fname)[0]
                           for fname in os.listdir(self.image_dir)]
        else:
            self.fnames = [os.path.splitext(fname)[0]
                           for fname in fnames]
        self.fnames = np.array(self.fnames)

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        if isinstance(index, int):
            fname = self.fnames[index]
            image = self._get_image(fname)
            label = self._get_label(fname)
            return image, label
        else:
            fnames = self.fnames[index]
            images = []
            labels = []
            for fname in fnames:
                images.append(self._get_image(fname))
                labels.append(self._get_label(fname))
            return np.stack(images), np.stack(labels)

    def _get_image(self, fname):
        image_path = os.path.join(self.image_dir, fname) + ".jpg"
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _get_label(self, fname):
        label_path = os.path.join(self.seg_dir, fname) + ".png"
        label = cv2.imread(label_path, 0)
        return label

    def draw_image(self, fname, alpha=0.3):
        if isinstance(fname, int):
            fname = self.fnames[fname]

        image = self._get_image(fname)
        label = self._get_label(fname)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        mask = label_to_color_image(label).astype(np.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2RGBA)
        return cv2.addWeighted(image, 1., mask, alpha, 0.)

    def shuffle(self):
        np.random.shuffle(self.fnames)


def convert_to_label_map(json_string,
                         image_shape=(1080, 1920, 3),
                         ignore_values=(),
                         ignore_width=11):
    global LABEL_MAP
    blank = np.zeros(image_shape[:2], dtype=np.uint8)
    ignore = np.zeros(image_shape[:2], dtype=np.uint8)

    shapes = sorted(json_string['shapes'],
                    key=lambda x: LABEL_MAP[x['label']])
    for shape in shapes:
        label = LABEL_MAP[shape['label']]

        if label in ignore_values:
            continue

        pts = list([np.array(shape['points'], np.int32)])
        cv2.fillPoly(blank, pts, label)

        # Ignore 부분 계산
        temp = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.fillPoly(temp, pts, label)
        kernel = np.ones((3, 3), dtype=np.uint8)
        out = cv2.erode(temp.copy(), kernel, iterations=ignore_width)
        ignore = cv2.add(ignore, out - temp)
    ignore[ignore > 0] = 255

    return cv2.add(blank, ignore)


def convert_to_crack_maps(crack_df,
                          image_shape=(1080, 1920, 3),
                          ignore_width=0):
    for filename, subset_df in tqdm(crack_df.groupby('filename')):
        blank = np.zeros(image_shape[:2], dtype=np.uint8)
        ignore = np.zeros(image_shape[:2], dtype=np.uint8)

        if np.any(np.isnan(subset_df.segmentation.values[0])):
            pts = []
        else:
            pts = [pts.astype(np.int32)
                   for pts in subset_df.segmentation.values]
        cv2.fillPoly(blank, pts, 1)

        # Ignore 부분 계산
        temp = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.fillPoly(temp, pts, 1)
        kernel = np.ones((3, 3), dtype=np.uint8)
        out = cv2.erode(temp.copy(), kernel, iterations=ignore_width)
        ignore = cv2.add(ignore, out - temp)

        ignore[ignore > 0] = 255
        result = cv2.add(blank, ignore)

        save_path = os.path.join(ABSEG_DIR,filename)
        save_path = save_path.replace(".jpg", '.png')

        cv2.imwrite(save_path, result)


def read_crack_df(abnormal_path):
    with open(abnormal_path, 'r') as f:
        annotation_dict = json.load(f)

    # Image 이름과 Image ID 간 정보
    f_df = pd.DataFrame(annotation_dict['images'])
    f_df = f_df.rename(columns={"file_name": "filename"})
    f_df = f_df[['filename', 'id']]
    f_df.rename(columns={'id': 'image_id'}, inplace=True)

    anno_df = pd.DataFrame(annotation_dict['annotations'])

    anno_df.bbox = (anno_df.bbox
                    .map(lambda xs: np.array(xs).astype(np.int)))
    anno_df.segmentation = (anno_df.segmentation
                            .map(lambda xs: np.array(xs).astype(np.int)))
    anno_df.segmentation = (anno_df.segmentation
                            .map(lambda xs: xs.reshape(-1, 2)))

    return pd.merge(f_df, anno_df, how='left', on='image_id')

