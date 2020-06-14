"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
"""
학습코드 예시입니다.
"""
import tensorflow as tf
import numpy as np
import os, sys
import argparse
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from keras_tqdm import TQDMCallback
from datetime import datetime

np.random.seed(1)
assert int(tf.__version__[0] ) >= 2., "Tensorflow 2.0 버전위에서 동작 가능합니다."

# Project Path 설정
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR,"datasets/")
CRACK_DIR = os.path.join(DATA_DIR,"pavement_crack/")
LOG_DIR = os.path.join(ROOT_DIR,"logs/")
os.makedirs(LOG_DIR, exist_ok=True)

sys.path.append(ROOT_DIR)
from models.generator import CrackGenerator
from models.config import  ModelConfig
from models.model import DeepLabV3
from models.optimizer import AdamW
from models.losses import sparse_categorical_crossentropy_with_ignore
from models.losses import binary_crossentropy_with_ignore
from models.metrics import mean_iou, binary_iou


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-test', action="store_true", default=False)
    args = parser.parse_args()
    test_flag = args.test

    # 모델 가져오기
    config = ModelConfig()
    config.SEPARATE_BG_CLASSIFIER = True
    model = DeepLabV3(config)

    # Optimizer 정하기
    optm = AdamW(lr=7e-4)

    # Loss Function 설정하기
    loss = {
        "bg_prediction":
        binary_crossentropy_with_ignore(ignore_value=255),
        "label_prediction" :
        sparse_categorical_crossentropy_with_ignore(ignore_value=255),
        "crack_prediction" :
        binary_crossentropy_with_ignore(ignore_value=255)}

    # Loss Weight 설정하기
    loss_weights = {
        "bg_prediction" : 1.0,
        "label_prediction" : 1.0,
        "crack_prediction" : 4.0
    }

    # Metric 함수 설정하기
    metrics={
         "bg_prediction" : binary_iou(thr=0.5),
         "label_prediction" : mean_iou,
         "crack_prediction" : binary_iou(thr=0.5)
    }

    # 모델 Compile 하기
    model.compile(optm, loss=loss,
                  loss_weights=loss_weights,
                  metrics=metrics)

    # Train Valid Set 구분하기
    fnames = os.listdir(os.path.join(CRACK_DIR, "images/"))
    np.random.shuffle(fnames)

    valid_ratio = 0.1 # Validation Dataset 비율
    valid_nums = int(len(fnames) * valid_ratio)

    valid_fnames = fnames[:valid_nums]
    train_fnames = fnames[valid_nums:]
    if test_flag:
        valid_fnames = valid_fnames[:8]
        train_fnames = train_fnames[:8]


    # Generator 구성하기
    crop_ratio = 0.7
    image_shape = (1080, 1920, 3)

    crop_size = (np.array(image_shape[:2])*crop_ratio).astype(np.int)

    traingen = CrackGenerator(CRACK_DIR,
                              train_fnames,
                              batch_size=8,
                              use_crack=True,
                              use_background=True,
                              crop_size=crop_size,
                              gamma_ratio=(0.5, 1.5),
                              random_crop=True,
                              random_flip=True,
                              random_gamma=True,
                              random_noise=True,
                              random_scale=True)

    validgen = CrackGenerator(CRACK_DIR,
                              valid_fnames,
                              batch_size=4,
                              use_crack=True,
                              use_background=True,
                              random_crop=False,
                              random_flip=False,
                              random_gamma=False,
                              random_noise=False,
                              random_scale=False,
                              shuffle=False)

    # CallBacks setting하기
    callbacks = []

    # Early Stopping
    callbacks.append(EarlyStopping(monitor='val_loss',
                                   patience=5,
                                   verbose=0))
    dirname = datetime.now().strftime("%m%d-%H%M")
    save_dir = os.path.join(LOG_DIR,dirname)
    os.makedirs(save_dir,exist_ok=True)

    # Model Checkpoint
    ckpt_path = os.path.join(save_dir,
                             "weights.{val_loss:.2f}-{epoch:02d}.hdf5")
    callbacks.append(ModelCheckpoint(ckpt_path,
                                     save_best_only=True, save_weights_only=True))
    with open(os.path.join(save_dir,"model.json"),'w') as f:
        f.write(model.to_json())

    #CSV Logger
    callbacks.append(CSVLogger(os.path.join(save_dir, "training.log")))

    #TQDM callback
    cb = TQDMCallback()
    setattr(cb, 'on_train_batch_begin', lambda *x: None)
    setattr(cb, 'on_train_batch_end', lambda *x: None)
    setattr(cb, 'on_test_batch_begin', lambda *x: None)
    setattr(cb, 'on_test_batch_end', lambda *x: None)
    setattr(cb, 'on_test_begin', lambda *x: None)
    setattr(cb, 'on_test_end', lambda *x: None)
    callbacks.append(cb)

    print(f"SAVE Directory : {save_dir}")
    sys.stdout.flush()
    if test_flag:
        hist = model.fit_generator(traingen,
                                   epochs=2,
                                   validation_data=validgen,
                                   callbacks=callbacks,
                                   use_multiprocessing=True,
                                   workers=5)
    else:
        hist = model.fit_generator(traingen,
                                   epochs=30,
                                   validation_data=validgen,
                                   callbacks=callbacks,
                                   use_multiprocessing=True,
                                   workers=5)
