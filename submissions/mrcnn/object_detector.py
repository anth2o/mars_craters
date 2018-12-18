import skimage
import numpy as np
import cv2
import imgaug
import tensorflow as tf
import math
import os
 
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
 
class ObjectDetector:
    def __init__(self, epoch_head=1, epoch_all=6, collab=False):
        self.height=256
        self.width=256
 
        self.epoch_head=epoch_head
        self.epoch_all=epoch_all
 
        self.config=CraterConfig()
        self.model=modellib.MaskRCNN(mode = "training", model_dir = "logs",
                                config = self.config)
        self.collab = collab
 
    @staticmethod
    def split_train_test(X, y, valid_ratio = 0.1):
        nb_examples=len(X)
        nb_valid=int(valid_ratio * nb_examples)
        nb_train=nb_examples - nb_valid
        indices=np.arange(nb_examples)
        train_indices=indices[0: nb_train]
        valid_indices = indices[nb_train:]
 
        X_train=X[train_indices]
        y_train=y[train_indices]
        X_valid=X[valid_indices]
        y_valid=y[valid_indices]
 
        return X_train, y_train, X_valid, y_valid
 
    def fit(self, X, y):
        COCO_MODEL_PATH = "mask_rcnn_coco.h5"
        if not os.path.exists(COCO_MODEL_PATH):
            utils.download_trained_weights(COCO_MODEL_PATH)
        self.model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
 
        X_train, y_train, X_valid, y_valid=self.split_train_test(X, y)
 
        # Training dataset
        dataset_train=CratersDataset(data=X_train, labels=y_train, height=self.height, width=self.width)
        dataset_train.load_craters()
        dataset_train.prepare()
 
        print("Image Count: {}".format(len(dataset_train.image_ids)))
        print("Class Count: {}".format(dataset_train.num_classes))
 
        for i, info in enumerate(dataset_train.class_info):
            print("{:3}. {:50}".format(i, info['name']))
 
        # Testing dataset
        dataset_val=CratersDataset(data=X_valid, labels=y_valid, height=self.height, width=self.width)
        dataset_val.load_craters()
        dataset_val.prepare()

        # Personal add : Add augmentation 
        augmentation=imgaug.augmenters.Sometimes(0.5, [
            imgaug.augmenters.Fliplr(0.5),
            imgaug.augmenters.Flipud(0.5),
            imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0)),
            imgaug.augmenters.Multiply(0.75),
            imgaug.augmenters.Multiply(1.25),
        ])
 
        # Train the head branches
        # Passing layers="heads" freezes all layers except the head
        # layers. You can also pass a regular expression to select
        # which layers to train by name pattern.
 
        print("Training network heads")
 
        # Fine tune all layers
        # Passing layers="all" trains all layers. You can also
        # pass a regular expression to select which layers to
        # train by name pattern.
        if self.collab:
            self.model = tf.contrib.tpu.keras_to_tpu_model(
                self.model,
                strategy=tf.contrib.tpu.TPUDistributionStrategy(
                    tf.contrib.cluster_resolver.TPUClusterResolver(
                        tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
                )
)
        self.model.train(dataset_train, dataset_val,
                    learning_rate=self.config.LEARNING_RATE / 5.,
                    epochs=self.epoch_all,
                    layers="all",
                    augmentation=augmentation)
 
    def predict(self, X):
        inference_config=InferenceConfig()
 
        # Recreate the model in inference mode
        self.model=modellib.MaskRCNN(mode = "inference",
                                  config= inference_config,
                                  model_dir = "logs")
 
        # Get path to saved weights
        # Either set a specific path or find last trained weights
        # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
        model_path=self.model.find_last()
 
        # Load trained weights
        print("Loading weights from ", model_path)
        self.model.load_weights(model_path, by_name = True)
 
        num_images=len(X)
        y_pred=[]
 
        for i in range(num_images):
            img=self._resize_image(image=X[i], shape=(self.height, self.width))
            pred=self.model.detect([img], verbose = 0)[0]
            bboxes=pred['rois']
            confs=pred['scores']
 
            # confs = self.recalibrate_confs(confs)
            circles_pred=self._anchor_to_circle(bboxes, confs)
            y_pred.append([tuple(circles_pred[i]) for i in range(len(circles_pred))])
 
        y_pred_array=np.empty(len(y_pred), dtype = object)
        y_pred_array[:] = y_pred
 
        return y_pred_array
 
    @staticmethod
    def _anchor_to_circle(bbox, confs):
        """Convert the bouding box predicted to circlular prediction.
 
        Parameters
        ----------
        bbox : list
            Each tuple is organized as [conf, x_min, x_max, y_min, y_max]
 
        Returns
        -------
        circles : list of tuples
            Each tuple is organized as [conf, cx, cy, radius].
 
        """
        res=[]
 
        for i in range(len(bbox)):
            x_min, y_min, x_max, y_max=bbox[i]
            radius=224 * ((((x_max - x_min) + (y_max - y_min)) / 2) / 2) / 256
            cx=224 * (x_min + (x_max - x_min) / 2) / 256
            cy=224 * (y_min + (y_max - y_min) / 2) / 256
            # Personal add : The more the box is different of a circle, the less the confidence is
            conf_mult = 2 * math.sqrt((y_max - y_min) * (x_max - x_min)) / (y_max - y_min + x_max - x_min)
            if radius >= 5 and radius <= 28:
                res.append([confs[i]*conf_mult, cx, cy, radius])
        return np.array(res)
 
    @staticmethod
    def _resize_image(image, shape):
        """
            Resize image using to the new specified width.
        """
 
        # resize the image
        resized=cv2.resize(image, shape)
        # If grayscale. Convert to RGB for consistency.
        if len(resized.shape) != 3:
            resized=skimage.color.gray2rgb(resized)
 
        # return the resized image
        return resized
 
 
############################################################
#  Configuration class
############################################################
 
class CraterConfig(Config):
    """Configuration for training on the mars craters dataset.
    Derives from the base Config class and overrides values specific
    to the mars craters dataset.
    """
    # Give the configuration a recognizable name
    NAME = "craters"
 
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8
 
    # Feature extractor
    BACKBONE = 'resnet50'
 
    # Number of classes (including background)
    NUM_CLASSES = 2  # background + crater
 
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
 
    RPN_ANCHOR_SCALES = (5, 8, 10, 15, 22)  # anchor side in pixels
 
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 64
 
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 750
 
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50
 
    DETECTION_MIN_CONFIDENCE = 0.95
    LEARNING_RATE = 0.01

    CHECKPOINT_PATH = 'mrcnn_weights.h5'
 
 
############################################################
#  Dataset class
############################################################
 
class CratersDataset(utils.Dataset):
    """
        The dataset consists of images of mars craters.
    """
 
    def __init__(self, data, labels, height, width):
        super().__init__(self)
        # Add classes. We have only one class to add.
        self.add_class("crater", 1, "crater")
        self.data = data
        self.labels = labels
        self.height = height
        self.width = width
        self.img_height = data[0].shape[0]
        self.img_width = data[0].shape[1]
 
    def load_craters(self):
        for i in range(len(self.data)):
            self.add_image("crater", image_id=i, width=self.width, height=self.height, label=self.labels[i], path=None)
 
    def load_image(self, image_id):
        """
            Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image_info = self.image_info[image_id]
        shape = (image_info['height'], image_info['width'])
 
        img_resized = self._resize_image(self.data[image_id], shape)
 
        return img_resized
 
    @staticmethod
    def _resize_image(image, shape):
        """
            Resize image using to the new specified width.
        """
 
        # resize the image
        resized = cv2.resize(image, shape)
        # If grayscale. Convert to RGB for consistency.
        if len(resized.shape) != 3:
            resized = skimage.color.gray2rgb(resized)
 
        # return the resized image
        return resized
 
    def rescale_craters(self, craters, new_height):
        return np.multiply(craters, (new_height / self.img_height))
 
    def load_mask(self, image_id):
        """
            Generate instance masks for an image.
 
            Returns:
            -------
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        # craters = self.rescale_craters(craters=image_info['label'], new_height=image_info['height'])
        craters = image_info['label']
        nb_craters = len(craters)
 
        if nb_craters == 0:
            mask = np.zeros([image_info["height"], image_info["width"], 1], dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros([image_info["height"], image_info["width"], nb_craters], dtype=np.uint8)
            class_ids = np.zeros((nb_craters,), dtype=np.int32)
            n=self.height
 
            for i in range(nb_craters):
                label=craters[i]
                # Get coordinates of the labeled circle
                coord_center_y = 256 * label[0] / 224
                coord_center_x = 256 * label[1] / 224
                r = 256 * label[2] / 224
 
                # get coordinates of pixels in the circle
                y, x=np.ogrid[-coord_center_x: n -
                                coord_center_x, -coord_center_y: n - coord_center_y]
                temp_mask=x * x + y * y <= r * r
                x_points_mask, y_points_mask=np.where(temp_mask == True)
 
                # set the pixels inside the circle to 1
                mask[y_points_mask, x_points_mask, i]=1
 
                class_ids[i] = 1
 
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), class_ids.astype(np.int32)
 
 
############################################################
#  Inference model
############################################################
 
class InferenceConfig(CraterConfig):
    GPU_COUNT=1
    IMAGES_PER_GPU=1