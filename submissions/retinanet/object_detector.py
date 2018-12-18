from __future__ import division

from math import ceil
import os
import shutil


import keras
import keras_retinanet.losses
import numpy as np
import pandas as pd
import scipy.misc
from keras.callbacks import (Callback, EarlyStopping, ModelCheckpoint,
                             ReduceLROnPlateau)
from keras.layers import Input
from keras.optimizers import SGD, Adam, adam
from keras.preprocessing.image import ImageDataGenerator
from keras_retinanet.losses import focal
from keras_retinanet.models import backbone
from keras_retinanet.models.resnet import resnet50_retinanet
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.preprocessing.generator import Generator
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

from keras_retinanet import models

from PIL import Image
from sklearn.utils import Bunch
from ssd_keras.ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y


class ObjectDetector(object):
    rescale_parameter = 0.4
    min_radius = 5
    max_radius = 28
    rot_matrix = [np.array([[1, 0], [0, 1]]), np.array([[0, 1], [-1, 0]]), np.array([[-1, 0], [0, -1]]), np.array([[0, -1], [1, 0]])]

    def __init__(self, batch_size=32, epoch=1, model_check_point=True, flip_images=True, rotate_images=True, change_brightness=True, to_delete=True):
        self.model_ = self._build_model()
        if to_delete:
            # model_path = os.path.join('.', 'submissions', 'retinanet', 'resnet50_coco_best_v2.1.0.h5')
            # model = models.load_model(model_path, backbone_name='resnet50')
            try:
                shutil.rmtree('./data/img/train')
            except Exception as e:
                print(e)
            try:
                shutil.rmtree('./data/img/test')
            except Exception as e:
                print(e)
        self.batch_size = batch_size
        self.epoch = epoch
        self.model_check_point = model_check_point
        self.flip_images = flip_images
        self.rotate_images = rotate_images
        self.change_brightness = change_brightness
        
    def _build_model(self):
        model = backbone('resnet50').retinanet(num_classes=1)
        adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-05)
        model.compile(
            optimizer=adam,
            loss={
                'regression'    : keras_retinanet.losses.smooth_l1(),
                'classification': keras_retinanet.losses.focal()
            })
       return model

    def fit(self, X, y):
        print(os.getcwd())
        train_dataset = BatchGeneratorBuilder(X, y, self.flip_images, self.rotate_images, self.change_brightness)
        train_generator, val_generator, n_train_samples, n_val_samples = train_dataset.get_train_valid_generators(batch_size=self.batch_size)
        
        # create the callbacks to get during fitting
        callbacks = []
        if self.model_check_point:
            callbacks.append(
                ModelCheckpoint('./retinanet_weights_best.h5',
                                monitor='val_loss', verbose=1,
                                save_best_only=True, save_weights_only=True,
                                mode='auto', period=1))
        # add early stopping
        callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.001,
                                       patience=10, verbose=1))

        # reduce learning-rate when reaching plateau
        callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                           patience=5, epsilon=0.001,
                                           cooldown=2, verbose=1))
        # fit the model
        self.model_.fit_generator(
            generator=train_generator,
            steps_per_epoch=ceil(n_train_samples / self.batch_size),
            epochs=self.epoch,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=ceil(n_val_samples / self.batch_size))

    @staticmethod
    def _anchor_to_circle(boxes, pred=True):
        res = []
        for box in boxes:
            if pred:
                box = box[1:]
            conf, x_min, x_max, y_min, y_max = box
            radius = (((x_max - x_min) + (y_max - y_min)) / 2) / 2
            cx = x_min + (x_max - x_min) / 2
            cy = y_min + (y_max - y_min) / 2
            if pred:
                # filter crater based on their radius
                if radius >= ObjectDetector.min_radius and radius <= ObjectDetector.max_radius:
                    conf *= 0.5 / ObjectDetector.rescale_parameter
                    res.append((conf, cy, cx, radius))
            else:
                res.append((cy, cx, radius))
        return res

    def predict(self, X):
        try:
            os.mkdir('./data/img/test')
        except Exception as e:
            print(e)
        for i in range(len(X)):
            im = Image.fromarray(X[i])
            im.save('./data/img/test/'+str(i)+'.jpg')

        self.model_converted = models.convert_model(self.model_)

        self.boxes_list = []
        self.scores_list = []
        self.labels_list = []

        for i in range(len(X)):
            image = read_image_bgr('./data/img/test/'+str(i)+'.jpg')
            boxes, scores, labels = self.model_converted.predict_on_batch(np.expand_dims(image, axis=0))
            # indices_craters = np.argwhere(labels==1.0).T[0]
            # boxes = boxes[:, indices_craters]
            # scores = scores[:, indices_craters]
            self.boxes_list.append(boxes)
            self.scores_list.append(scores)
            self.labels_list.append(labels)
        

class BatchGeneratorBuilder(object):
    def __init__(self, X_array, y_array=None, flip_images=True, rotate_images=True, change_brightness=True, train=True):
        self.X_array = X_array
        self.y_array = y_array
        self.nb_examples = len(X_array)
        self.flip_images = flip_images
        self.rotate_images = rotate_images
        self.change_brightness = change_brightness
        self.column_label = {
            'col_filename': 'filename',
            'col_label': 'label',
            'col_x1': 'x1',
            'col_y1': 'y1',
            'col_x2': 'x2',
            'col_y2': 'y2'
        }
        if train:
            self.img_directory = './data/img/train'
        else:
            self.img_directory = './data/img/test'

    def get_train_valid_generators(self, batch_size=256, valid_ratio=0.1):
        print('Generating dataset')
        try:
            os.mkdir(self.img_directory)
        except Exception as e:
            print(e)
        for i in range(len(self.X_array)):
            im = Image.fromarray(self.X_array[i])
            im.save(self.img_directory+'/'+str(i)+'.jpg')
        nb_valid = int(valid_ratio * self.nb_examples)
        nb_train = self.nb_examples - nb_valid
        indices = np.arange(self.nb_examples)
        train_indices = indices[0:nb_train]
        valid_indices = indices[nb_train:]
        gen_train = self._get_generator(
            indices=train_indices, batch_size=batch_size)
        gen_valid = self._get_generator(
            indices=valid_indices, batch_size=batch_size)
        return gen_train, gen_valid, nb_train, nb_valid
                
    def _get_generator(self, indices=None, batch_size=256):
        if indices is None:
            indices = np.arange(self.nb_examples)
        df = pd.DataFrame(columns=['filename', 'label', 'x1', 'y1', 'x2', 'y2'])
        j = 0
        for i in indices:            
            for (cy, cx, r) in self.y_array[i]:
                filename = self.img_directory+ '/' + str(i) + '.jpg'
                df.loc[j] = [filename, 'crater', max(0.0, cx - r), max(cy - r, 0.0), min(cx + r, 224.0), min(cy + r, 224.0)]
                j += 1
        return DfGenerator(df, {'crater': 0}, self.column_label)
                
    def _process_batch(self, X_batch, y_batch):
        # flip images
        if self.flip_images:
            if np.random.randint(2):
                X_batch = np.flip(X_batch, axis=0)
                y_batch = [(224 - row, col, radius)
                              for (row, col, radius) in y_batch]
            if np.random.randint(2):
                X_batch = np.flip(X_batch, axis=1)
                y_batch = [(row, 224 - col, radius)
                              for (row, col, radius) in y_batch]

        # rotating images
        if self.rotate_images:
            rand_rotation = np.random.randint(4)
            X_batch = np.rot90(X_batch, rand_rotation)
            for k in range(len(y_batch)):
                y_temp = y_batch[k]
                coord = ObjectDetector.rot_matrix[rand_rotation] @ np.array([y_temp[1] - 112, y_temp[0] - 112]) + np.array([112, 112])
                y_batch[k] = (coord[1], coord[0], y_temp[2])

        # increase or decrease brightness
        if self.change_brightness:
            rand_brightness = np.random.rand()
            max_X = X_batch.max()
            X_batch = np.round((X_batch*1.0/max_X)**(0.25 + 1.5*rand_brightness) * max_X).astype(int)

        return X_batch, y_batch
                

class DfGenerator(CSVGenerator):
    """Custom generator intented to work with in-memory Pandas' dataframe."""
    def __init__(self, df, class_mapping, cols, base_dir='', **kwargs):
        """Initialization method.
        Arguments:
            df: Pandas DataFrame containing paths, labels, and bounding boxes.
            class_mapping: Dict mapping label_str to id_int.
            cols: Dict Mapping 'col_{filename/label/x1/y1/x2/y2} to corresponding df col.
        """
        self.base_dir = base_dir
        self.cols = cols
        self.classes = class_mapping
        self.labels = {v: k for k, v in self.classes.items()}

        self.image_data = self._read_data(df)
        self.image_names = list(self.image_data.keys())

        Generator.__init__(self, **kwargs)

    def _read_classes(self, df):
        return {row[0]: row[1] for _, row in df.iterrows()}

    def __len__(self):
        return len(self.image_names)

    def _read_data(self, df):
        data = {}
        for _, row in df.iterrows():
            img_file, class_name = row[self.cols['col_filename']], row[self.cols['col_label']]
            x1, y1 = row[self.cols['col_x1']], row[self.cols['col_y1']]
            x2, y2 = row[self.cols['col_x2']], row[self.cols['col_y2']]
            if img_file not in data:
                data[img_file] = []
            if not isinstance(class_name, str) and np.isnan(class_name):
                continue
            data[img_file].append({
                'x1': int(x1), 'x2': int(x2),
                'y1': int(y1), 'y2': int(y2),
                'class': class_name
            })

        return data
