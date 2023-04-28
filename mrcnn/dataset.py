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

import utils
import addon_utils


##############################
##     GLOBAL VARS
##############################
from mrcnn import logger

##############################
##     DATASET CLASS
##############################
class Dataset:
	"""
		Dataset class for VGG Image Annotator. Read images, apply augmentation and preprocessing transformations.
			Args:
				images_dir:           (str): path to images folder
				class_key:            (str): class_key may be a key for class name for polygons
				augmentation:         (albumentations.Compose): data transfromation pipeline
				preprocess_transform: (albumentations.Compose):  transformation of an image
				json_annotation_key:  (str): default key to extract annotations from .json. By default, it is '_via_img_metadata' for VGG Image Annotator
				**kwargs:             additional processing configuration parameters
	"""
	def __init__(self, 
		preprocessor=None,
		augmenter=None,
		verbose: bool = False, 
		**kwargs
	):
		super(Dataset, self).__init__()	
		
		# - Set args
		self.preprocessor= preprocessor
		self.augmenter= augmenter
		self.kwargs = kwargs
		self.verbose = verbose
		
		# - Init variables
		self.loaded_imgs= 0
		self.nobjs_per_class= {}
		self.image_info = []
				
		# - Get class indexes from class_dict
		self.classes_dict = self.kwargs['class_dict']
		self.nclasses= len(self.classes_dict)
		self.class_values = list(self.classes_dict.values())
		
		# - Compute backbon shapes & anchors
		self.backbone_shapes = utils.compute_backbone_shapes(self.kwargs)
		self.anchors = utils.generate_pyramid_anchors(
			scales=self.kwargs['rpn_anchor_scales'],
			ratios=self.kwargs['rpn_anchor_ratios'],
			feature_shapes=self.backbone_shapes,
			feature_strides=self.kwargs['backbone_strides'],
			anchor_stride=self.kwargs['rpn_anchor_stride']
		)
		if self.verbose:
			print(f'\nBackbone shapes: {self.backbone_shapes}')
			print(f'\nAnchors: {self.anchors.shape}')



	# ================================================================
	# ==   SET PROPERTIES
	# ================================================================
	def set_class_dict(self, class_dict_str):
		""" Set class dictionary from json string """

		# - Check
		if class_dict_str=="":
			logger.error("Empty string given!")
			return -1

		# - Set class id dictionary
		logger.info("Set class id dictionary ...")
		class_dict= {}
		try:
			class_dict= json.loads(class_dict_str)
		except Exception as e:
			logger.error("Failed to load class dictionary from string (err=%s)!" % (str(e)))
			return -1
			
		self.classes_dict= class_dict
		
		# - Set number of classes
		self.nclasses= len(self.classes_dict)
		self.class_values = list(self.classes_dict.values())
		
		logger.debug("classes_dict=%s, nclasses=%d" % (str(self.classes_dict), self.nclasses))
	
		return 0

	# ================================================================
	# ==   ADD/LOAD IMAGE
	# ================================================================
	def add_image(self, image_id, path, path_masks, class_ids, **kwargs):
		""" Add image to dataset """

		image_info = {
			"id": image_id,
			"path": path,
			"path_masks": path_masks,
			"class_ids": class_ids
		}
		image_info.update(kwargs)
		self.image_info.append(image_info)

	# ================================================================
	# ==   LOAD DATASET FROM JSON
	# ================================================================
	def load_data_from_json_file(self, filename, rootdir='', modify_class_names=True):
		""" Load dataset specified in a json file """

		# - Read json file
		try:
			json_file= open(filename,"r")
		except IOError:
			logger.error("Failed to open file %s!" % (filename))
			return -1
	
		# - Read obj info
		try:
			d= json.load(json_file)
		except Exception as e:
			logger.error("Failed to load json file %s (err=%s)!" % (filename, str(e)))
			return -1
			
		# - Parse file info
		img_path= os.path.join(rootdir, d['img'])
		img_fullpath= os.path.abspath(img_path)
		img_path_base= os.path.basename(img_fullpath)
		img_path_base_noext= os.path.splitext(img_path_base)[0]
		img_id= str(uuid.uuid1())
	
		logger.debug("img_path=%s, img_fullpath=%s" % (img_path,img_fullpath))

		valid_img= (os.path.isfile(img_fullpath) and img_fullpath.endswith('.fits'))
		if not valid_img:
			logger.warn("Image file %s does not exist or has unexpected extension (.fits required)!" % (img_fullpath))
			return -1
			
		# - Read image metadata
		img_metadata= {}
		img_metadata["telescope"]= d["telescope"]
		img_metadata["bkg"]= d["bkg"]
		img_metadata["rms"]= d["rms"]
		img_metadata["bmaj"]= d["bmaj"]
		img_metadata["bmin"]= d["bmin"]
		img_metadata["dx"]= d["dx"]
		img_metadata["dy"]= d["dy"]
		img_metadata["nx"]= d["nx"]
		img_metadata["ny"]= d["ny"]

		# - Read object info
		nobjs= len(d['objs'])
		logger.debug("#%d objects present in file %s ..." % (nobjs, filename))
				
		mask_paths= []
		class_ids= []
		sidelobes_mixed_or_near= []
		good_masks= True
				
		for obj_dict in d['objs']:
			mask_path= os.path.join(rootdir,obj_dict['mask'])
			mask_fullpath= os.path.abspath(mask_path)
			valid_img= (os.path.isfile(mask_fullpath) and mask_fullpath.endswith('.fits'))
			if not valid_img:
				good_masks= False
				break

			is_flagged= obj_dict['sidelobe-mixed']
			nislands= obj_dict['nislands']
			class_name= obj_dict['class']
	
			# - Use multi-island and flagged classes?
			if modify_class_names:
				if nislands>1 and class_name=="extended":
					class_name= 'extended-multisland'
				if is_flagged:
					class_name= 'flagged'
				obj_dict['class']= class_name

			# - Get class ID from name
			class_id= 0
			if class_name in self.classes_dict:
				class_id= self.classes_dict.get(class_name)
			else:
				logger.warn("Class name %s for image file %s is not present in dictionary, skip this object ..." % (class_name, img_fullpath))
				continue			

			# - Check if this mask is that of a source close to a sidelobe
			sidelobe_mixed_or_near = 0
			if ('sidelobe-mixed' in obj_dict) and ('sidelobe-near' in obj_dict):
				if (obj_dict['sidelobe-mixed'] == 1) or (obj_dict['sidelobe-near'] == 1):
					sidelobe_mixed_or_near = 1

			mask_paths.append(mask_fullpath)
			class_ids.append(class_id)
			sidelobes_mixed_or_near.append(sidelobe_mixed_or_near)
				
		if not good_masks:
			logger.error("One or more mask of file %s does not exist or have unexpected extension (.fits required)" % img_fullpath)
			return -1
					
		# - Add image & mask informations in dataset class
		self.add_image(
			image_id=img_id,
			path=img_fullpath,
			path_masks=mask_paths,
			class_ids=class_ids,
			sidelobes_mixed_or_near=sidelobes_mixed_or_near,
			objs=d['objs'],
			metadata=img_metadata
		)

		# - Count number of objects per class
		for class_id in class_ids:
			self.nobjs_per_class[class_id]+= 1
	
		return 0
		
	# ================================================================
	# ==   LOAD DATASET FROM JSON FILELIST (row format: jsonfile)
	# ================================================================
	def load_data_from_json_list(self, filelist, nmaximgs):
		""" Load dataset specified in a json filelist """
	
		# - Read json filelist
		img_counter= 0
		status= 0

		with open(filelist,'r') as f:
			for filename in f:
				filename = filename.strip()
				logger.info("Loading dataset info from file %s ..." % (filename))

				# - Check if absolute path
				rootdir= ''
				if os.path.isabs(filename):
					rootdir= os.path.dirname(filename)

				# - Load from json file
				status= self.load_data_from_json_file(filename,rootdir)
				if status<0:
					logger.warn("Failed to load data for file %s (see logs), skip it ..." % (filename))
					continue

				img_counter+= 1
				self.loaded_imgs+= 1	
				if nmaximgs!=-1 and img_counter>=nmaximgs:
					logger.info("Max number (%d) of desired images reached, stop loading ..." % nmaximgs)
					break

		if status<0:
			logger.warn("One or more files have been skipped...")
		if img_counter<=0:
			logger.error("All files in list have been skipped!")		
			return -1
		logger.info("#%d images added in dataset..." % img_counter)

		return 0

	# ================================================================
	# ==   IMAGE READ
	# ================================================================
	def load_image(self, image_id: int) -> np.ndarray:
		""" Read image from disk """
		
		filename= self.image_info[image_id]['path']
		res= addon_utils.read_fits(filename, strip_deg_axis=True)
		if res is None:
			logger.error("Failed to read image %s (id=%d) (see logs)!" % (filename, image_id))
			return None
		
		image= res[0]
		#header= res[1]
		#wcs= res[2]
		
		return image
		
	# ================================================================
	# ==   LOAD MASK (multiple objects per image)
	# ================================================================
	def load_mask(self, image_id: int) -> np.ndarray:
		""" Generate instance masks for an image.
				Returns:
					masks: A bool array of shape [height, width, instance count] with one mask per instance.
					class_ids: a 1D array of class IDs of the instance masks.
		"""

		# - Set bitmap mask of shape [height, width, instance_count]
		info = self.image_info[image_id]
		filenames= info["path_masks"]
		class_ids= info["class_ids"]
		nobjs= len(filenames)

		# - Read mask files
		mask = None	
		counter= 0

		for filename in filenames:
			res= addon_utils.read_fits(filename, strip_deg_axis=True)
			if res is None:
				logger.error("Failed to read mask file %s!" % (filename))
				return None
			data= res[0]
			height= data.shape[0]
			width= data.shape[1]
			data= data.astype(np.bool)
			if mask is None:
				mask = np.zeros([height,width,nobjs], dtype=np.bool)
			mask[:,:,counter]= data
			counter+= 1

		class_ids_array= np.full([mask.shape[-1]], class_ids, dtype=np.int32)
		#class_ids_array= np.array(class_ids, dtype=np.int32)
		
		# - Return mask, and array of class IDs of each instance
		return mask, class_ids_array
		
	
	def __len__(self):
		""" Return dataset size """
		return len(self.image_info)


	def __getitem__(self, idx: int):
		"""
			Generate item
				Args:
					idx: index of the image to read
				Returns: image, mask, bbox, image_meta, class_ids
		"""
        
		# - Read image 
		logger.debug("Read image ...")    
		image= self.load_image(idx)
		if original_image is None:
			logger.error("Failed to load image id %d!" % (idx))
			return None
			
		original_image= np.copy(image)
		original_image_shape = original_image.shape

		# - Apply pre-processing?
		if self.preprocessor is not None:
			logger.debug("Apply pre-processing ...")
			image= self.preprocessor(original_image)
			if image is None:
				logger.error("Failed to pre-process source image data at index %d (sname=%s, label=%s, classid=%s)!" % (index, sname, str(label), str(classid)))
				return None
				
		# - Read masks
		logger.debug("Read masks ...")
		res= self.load_mask(idx)
		if res is None:
			logger.error("Failed to load mask image id %d!" % (idx))
			return None
	
		original_masks_array= res[0]
		class_ids_array= res[1]
		
		# - Resize images
		logger.debug("Resize images ...")
		image, window, scale, padding, crop = utils.resize_image(
			image,
			min_dim=self.kwargs['image_min_dim'],
			min_scale=self.kwargs['image_min_scale'],
			max_dim=self.kwargs['image_max_dim'],
			mode=self.kwargs['image_resize_mode']
		)

		# - Resize masks
		logger.debug("Resize masks ...")
		masks_array = self.resize_mask(original_masks_array, scale, padding, crop)

		# - Apply augmentation
		_image_shape = image.shape

		if self.augmenter:
			masks_list = [masks_array[:, :, i].astype('float') for i in range(masks_array.shape[2])]
			transformed = self.augmenter(image=image, masks=masks_list)
			if transformed is None:
				logger.error("Failed to apply augmentation to images & masks!")
				retun None
			proc_image= transformed[0]
			proc_masks = transformed[1]

			assert proc_image.shape == _image_shape
			proc_masks = np.stack(proc_masks, axis=2)
		else:
			proc_image = image
			proc_masks = masks_array

		# - Note that some boxes might be all zeros if the corresponding mask got cropped out.
		#   and here is to filter them out
		_idx = np.sum(proc_masks, axis=(0, 1)) > 0
		proc_masks = proc_masks[:, :, _idx]
		proc_class_ids = class_ids_array[_idx]

		_orig_idx = np.sum(original_masks_array, axis=(0, 1)) > 0
		original_masks_array = original_masks_array[:, :, _orig_idx]
		original_class_ids = class_ids_array[_orig_idx]

		# - Compute bboxes
		bboxes = utils.extract_bboxes(proc_masks)
		original_bboxes = utils.extract_bboxes(original_masks_array)

		# - Active classes
		#   Different datasets have different classes, so track the classes supported in the dataset of this image.
		active_class_ids = np.zeros([len(self.classes_dict.keys())], dtype=np.int32)

		# 1 for classes that are in the dataset of the image
		# 0 for classes that are not in the dataset.
		# The position of ones and zeros means the class index.
		source_class_ids = list(self.classes_dict.values())  # self.classes_dict['licence'] or list(self.classes_dict.values())
		active_class_ids[source_class_ids] = 1

		# - Resize masks to smaller size to reduce memory usage
		if self.kwargs['use_mini_masks']:
			proc_masks = utils.minimize_mask(bboxes, proc_masks, self.kwargs['mini_mask_shape'])

		# - Image meta data
		image_meta = utils.compose_image_meta(idx, original_image_shape, window, scale, active_class_ids, self.kwargs)

		return proc_image, proc_masks, proc_class_ids, bboxes, image_meta, \
			original_image, original_masks_array, original_class_ids, original_bboxes


