#!/usr/bin/env python

from __future__ import print_function

##################################################
###          MODULE IMPORT
##################################################
## STANDARD MODULES
import os
import sys
import subprocess
import string
import time
import signal
from threading import Thread
import datetime
import numpy as np
import random
import math
import logging
from collections import Counter
import json
import warnings
from typing import Tuple
import uuid

# - Import TF modules
from tensorflow import cast
from tensorflow.keras.utils import Sequence

# - Import mrcnn modules
import utils
import addon_utils


##############################
##     GLOBAL VARS
##############################
from mrcnn import logger

##############################
##     DATA LOADER CLASS
##############################
class DataLoader(Sequence):
    """Load data from dataset and form batches

    Args:
        dataset:    Instance of Dataset class for image loading and preprocessing.
        detection_targets: bool, If True, generate detection targets (class IDs, bbox
                           deltas, and masks). Typically for debugging or visualizations because
                           in training detection targets are generated by DetectionTargetLayer.
        shuffle:     bool, if `True` shuffle image indexes each epoch.
        seed:        int, Seed for pseudo-random generator
        name:        str, DataLoader name
        cast_output: bool, Cast output to tensorflow.float32
        return_original: Return original images in batch
    """

    def __init__(self, dataset, detection_targets: bool = False, shuffle: bool = True, seed: int = 42,
                 name: str = 'dataloader', cast_output: bool = True, return_original: bool = False, **kwargs):

        self.seed = seed
        np.random.seed(self.seed)

        self.dataset = dataset
        self.random_rois = kwargs['random_rois']
        self.detection_targets = detection_targets
        self.indexes = np.arange(len(self.dataset))
        self.anchors = self.dataset.anchors
        self.backbone_shapes = self.dataset.backbone_shapes
        self.shuffle = shuffle
        self.cast_output = cast_output
        self.kwargs = kwargs
        self.batch_size = self.kwargs['batch_size']
        self.return_original = return_original

        self.on_epoch_end()

        self.name = name
        self.steps_per_epoch = self.__len__()
        print(f'\n{self.name} DataLoader. Steps per epoch: {self.steps_per_epoch}')

    def generate_batch(self, index: int):
        """
        Args:
            index: int to get an image

        Returns: python list
                    Example, input: (batch, w,h, 3):
                   'batch_images':       (batch, w, h, 3)
                   'batch_images_meta':  (batch, meta_shape)
                   'batch_rpn_match':    (batch, anchors, 1)
                   'batch_rpn_bbox':     (batch, rpn_train_anchors_per_image, 4)
                   'batch_gt_class_ids': (batch, max_gt_instances)
                   'batch_gt_boxes':     (batch, max_gt_instances, 4)
                   'batch_gt_masks':     (batch, w, h, max_gt_instances)
                   'batch_rpn_rois':     (batch, n_of_rois, 4), optional

        """
        # Set batch size counter
        gen_batch = 0
        while gen_batch < self.batch_size:

            image, gt_masks, gt_class_ids, gt_boxes, image_meta, \
            original_image, original_masks_array, original_class_ids, original_bboxes = self.dataset[index]
            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            if not np.any(gt_class_ids > 0):
                index = min(index + 1, len(self.indexes) - 1)
                continue

            # RPN Targets
            rpn_match, rpn_bbox = utils.build_rpn_targets(
                anchors=self.anchors,
                gt_class_ids=gt_class_ids,
                gt_boxes=gt_boxes,
                rpn_train_anchors_per_image=self.kwargs['rpn_train_anchors_per_image'],
                rpn_bbox_std=self.kwargs['rpn_bbox_std_dev']
            )

            # Mask R-CNN Targets
            if self.random_rois:
                rpn_rois = utils.generate_random_rois(image.shape, self.random_rois, gt_boxes)
                if self.detection_targets:
                    rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask = utils.build_detection_targets(
                        rpn_rois=rpn_rois, gt_class_ids=gt_class_ids, gt_boxes=gt_boxes, gt_masks=gt_masks,
                        train_rois_per_image=self.kwargs['train_rois_per_image'],
                        roi_pos_ratio=self.kwargs['roi_pos_ratio'],
                        num_classes=len(self.dataset.classes_dict.keys()),
                        bbox_std=self.kwargs['bbox_std'],
                        use_mini_mask=self.kwargs['use_mini_mask'],
                        mask_shape=self.kwargs['mask_shape'],
                        image_shape=self.kwargs['image_shape']
                    )

            # Init batch arrays
            if gen_batch == 0:
                batch_image_meta = np.zeros(
                    (self.batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_rpn_match = np.zeros(
                    [self.batch_size, self.anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros(
                    [self.batch_size, self.kwargs['rpn_train_anchors_per_image'], 4], dtype=rpn_bbox.dtype)
                batch_images = np.zeros(
                    (self.batch_size,) + image.shape, dtype=np.float32)
                batch_gt_class_ids = np.zeros(
                    (self.batch_size, self.kwargs['max_gt_instances']), dtype=np.int32)
                batch_gt_boxes = np.zeros(
                    (self.batch_size, self.kwargs['max_gt_instances'], 4), dtype=np.int32)
                batch_gt_masks = np.zeros(
                    (self.batch_size, gt_masks.shape[0], gt_masks.shape[1],
                     self.kwargs['max_gt_instances']), dtype=gt_masks.dtype)
                if self.random_rois:
                    batch_rpn_rois = np.zeros(
                        (self.batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
                    if self.detection_targets:
                        batch_rois = np.zeros(
                            (self.batch_size,) + rois.shape, dtype=rois.dtype)
                        batch_mrcnn_class_ids = np.zeros(
                            (self.batch_size,) + mrcnn_class_ids.shape, dtype=mrcnn_class_ids.dtype)
                        batch_mrcnn_bbox = np.zeros(
                            (self.batch_size,) + mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype)
                        batch_mrcnn_mask = np.zeros(
                            (self.batch_size,) + mrcnn_mask.shape, dtype=mrcnn_mask.dtype)

                if self.return_original:
                    batch_original_imgs = []
                    batch_original_masks = []
                    batch_original_class_ids = []
                    batch_original_bboxes = []

                    # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > self.kwargs['max_gt_instances']:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), self.kwargs['max_gt_instances'], replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]

            # Add to a batch
            batch_image_meta[gen_batch] = image_meta
            batch_rpn_match[gen_batch] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[gen_batch] = rpn_bbox
            batch_images[gen_batch] = image
            batch_gt_class_ids[gen_batch, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[gen_batch, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[gen_batch, :, :, :gt_masks.shape[-1]] = gt_masks
            if self.random_rois:
                batch_rpn_rois[gen_batch] = rpn_rois
                if self.detection_targets:
                    batch_rois[gen_batch] = rois
                    batch_mrcnn_class_ids[gen_batch] = mrcnn_class_ids
                    batch_mrcnn_bbox[gen_batch] = mrcnn_bbox
                    batch_mrcnn_mask[gen_batch] = mrcnn_mask

            if self.return_original:
                batch_original_imgs.append(original_image)
                batch_original_masks.append(original_masks_array)
                batch_original_class_ids.append(original_class_ids)
                batch_original_bboxes.append(original_bboxes)

            # Update info about batch size
            gen_batch += 1
            # Choose next index for the next image in batch or take the last image if one epoch is about to end.
            index = min(index + 1, len(self.indexes) - 1)

        inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                  batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
        outputs = []

        if self.random_rois:
            inputs.extend([batch_rpn_rois])
            if self.detection_targets:
                inputs.extend([batch_rois])
                # Keras requires that output and targets have the same number of dimensions
                batch_mrcnn_class_ids = np.expand_dims(batch_mrcnn_class_ids, -1)
                outputs.extend([batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask])

        if self.cast_output:
            inputs = [cast(x, addon_utils.set_type(x.dtype.name)) for x in inputs]
            outputs = [cast(x, addon_utils.set_type(x.dtype.name)) for x in outputs]

        if self.return_original:
            inputs.extend([batch_original_imgs, batch_original_masks, batch_original_class_ids, batch_original_bboxes])

        return inputs, outputs

    def __getitem__(self, i: int) -> Tuple[list, list]:
        inputs, outputs = self.generate_batch(i)
        return inputs, outputs

    def __len__(self):
        """
        Denotes the number of batches per epoch
        Returns:
        """
        return int(np.floor(len(self.indexes) / self.batch_size))

    def on_epoch_end(self):
        """
        Data shuffling after each epoch
        Returns: None
        """
        self.indexes = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.indexes)


