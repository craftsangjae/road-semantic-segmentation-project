"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
from tensorflow.keras.utils import Sequence
import os
import cv2
import numpy as np


class CrackGenerator(Sequence):
    "Generates Crack Semantic Segmentation Dataset for Keras"

    def __init__(self,
                 data_dir="pavement_crack/",
                 fnames=None, use_crack=True,
                 use_background=True, batch_size=32,
                 crop_size=(1000, 1000),
                 noise_scale=(0., 20.),
                 scale_ratio=(0.8, 1.2),
                 gamma_ratio=(0.6, 1.4),
                 random_noise=False,
                 random_scale=False,
                 random_gamma=False,
                 random_crop=False,
                 random_flip=False,
                 shuffle=True):
        "Initialization"
        if not os.path.exists(data_dir):
            raise ValueError("{}가 존재하지 않습니다.".format(data_dir))
        self.image_dir = os.path.join(data_dir, "images/")
        self.seg_dir = os.path.join(data_dir, "segmentations/")
        self.crack_dir = os.path.join(data_dir, "abnormal_segmentations/")
        if not fnames:
            self.fnames = [os.path.splitext(fname)[0]
                           for fname in os.listdir(self.image_dir)]
        else:
            self.fnames = [os.path.splitext(fname)[0]
                           for fname in fnames]
        self.fnames = np.array(self.fnames)
        # Arguments related to Y label
        self.use_crack = use_crack
        self.use_background = use_background
        self.batch_size = batch_size

        # Arguments related to Data Augmentation
        self.crop_size = crop_size
        self.noise_scale = noise_scale
        self.scale_ratio = scale_ratio
        self.gamma_ratio = gamma_ratio
        self.random_noise = random_noise
        self.random_scale = random_scale
        self.random_gamma = random_gamma
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return len(self.fnames) // self.batch_size

    def __getitem__(self, index):
        "Generator one batch of dataset"
        batch_fnames = self.fnames[self.batch_size * index:
                                   self.batch_size * (index + 1)]

        images, labels, bgs, cracks = [], [], [], []
        for fname in batch_fnames:
            image = self._get_image(fname)
            label = self._get_label(fname)

            if self.use_crack:
                crack = self._get_crack(fname)
                if crack is None:
                    crack = np.ones_like(label) * 255 # ignore value
            else:
                crack = 0

            image, label, crack = self._augment_data(image, label, crack)

            images.append(image)
            labels.append(label)
            cracks.append(crack)

        images = np.stack(images)
        labels = np.stack(labels)

        X = images
        Y = dict()
        if self.use_background:
            backgrounds = (labels==0).astype(labels.dtype)
            Y['bg_prediction'] = backgrounds
            labels[labels == 0] = 255 # ignore value
            Y['label_prediction'] = labels
        else:
            Y['label_prediction'] = labels

        if self.use_crack:
            cracks = np.stack(cracks)[:, :, :, None]
            Y['crack_prediction'] = cracks

        return X, Y

    def _get_image(self, fname):
        image_path = os.path.join(self.image_dir, fname) + ".jpg"
        image = cv2.imread(image_path)
        image = image[:, :, ::-1]
        return image

    def _get_label(self, fname):
        label_path = os.path.join(self.seg_dir, fname) + ".png"
        label = cv2.imread(label_path, 0)
        return label

    def _get_crack(self, fname):
        crack_path = os.path.join(self.crack_dir, fname) + ".png"
        if os.path.exists(crack_path):
            return cv2.imread(crack_path, 0)
        else:
            return None

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        if self.shuffle:
            np.random.shuffle(self.fnames)

    def _augment_data(self, image, label, crack):
        if self.random_scale:
            image, label, crack = self._random_scale(image, label, crack)
        if self.random_crop:
            image, label, crack = self._random_crop(image, label, crack)
        if self.random_flip:
            image, label, crack = self._random_flip(image, label, crack)
        if self.random_noise:
            image = self._random_noise(image)
        if self.random_gamma:
            image = self._random_gamma(image)

        return image, label, crack

    def _random_scale(self, image, label, crack):
        fx, fy = np.random.uniform(*self.scale_ratio, size=2)
        image = cv2.resize(image, None, fx=fx, fy=fy)
        label = cv2.resize(label, None, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
        crack = cv2.resize(crack, None, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
        return image, label, crack

    def _random_crop(self, *arrays):
        i_h, i_w = arrays[0].shape[:2]
        c_h, c_w = self.crop_size[:2]

        y = np.random.randint(0, i_h - c_h)
        x = np.random.randint(0, i_w - c_w)
        return [array[y:y + c_h, x:x + c_w] for array in arrays]

    def _random_flip(self, *arrays):
        if np.random.binomial(1, 0.5):
            return [array[:, ::-1] for array in arrays]
        else:
            return arrays

    def _random_noise(self, image):
        noise_strength = np.random.uniform(*self.noise_scale)
        noise = np.random.normal(0, noise_strength, size=image.shape)
        image = image.astype(np.float32) + noise
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def _random_gamma(self, image):
        gamma = np.random.uniform(*self.gamma_ratio)
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)
