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

## ASTROPY MODULES 
from astropy.io import ascii
from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip
from astropy.visualization import ZScaleInterval, MinMaxInterval, PercentileInterval, HistEqStretch

## SKIMAGE
from skimage.util import img_as_float64
from skimage.exposure import adjust_sigmoid, rescale_intensity, equalize_hist, equalize_adapthist

## IMG AUG
import imgaug
from imgaug import augmenters as iaa


##############################
##     GLOBAL VARS
##############################
from mrcnn import logger


#######################################
##     Custom Augmenters for imgaug
#######################################
class ZScaleAugmenter(iaa.meta.Augmenter):
	""" Apply ZScale transform to image as augmentation step """
	
	def __init__(self, 
		contrast=0.25, 
		random_contrast=False, 
		random_contrast_per_ch=False, contrast_min=0.1, contrast_max=0.7, 
		seed=None, name=None, random_state="deprecated", deterministic="deprecated"
	):
		""" Build class """

		# - Set parent class parameters
		super(ZScaleAugmenter, self).__init__(
			seed=seed, name=name,
			random_state=random_state, deterministic=deterministic
		)

		# - Set class parameters 
		if contrast<=0 or contrast>1:
			raise Exception("Expected contrast to be [0,1], got %f!" % (contrast))

		self.contrast= contrast
		self.random_contrast= random_contrast
		self.random_contrast_per_ch= random_contrast_per_ch
		self.contrast_min= contrast_min
		self.contrast_max= contrast_max
		self.seed= seed

	def get_parameters(self):
		""" Get class parameters """
		return [self.contrast, self.random_contrast, self.random_contrast_per_ch, self.contrast_min, self.contrast_max]

	def _augment_batch_(self, batch, random_state, parents, hooks):
		""" Augment batch of images """
	
		# - Check input batch
		if batch.images is None:
			return batch

		images = batch.images
		nb_images = len(images)
		contrasts= []

		# - Set random seed if given
		if self.seed is not None:
			np.random.seed(self.seed)

		# - Loop over image batch
		for i in range(nb_images):
			image= images[i]
			nb_channels = image.shape[2]

			# - Set zscale contrasts (fixed or random)
			if not contrasts:
				if self.random_contrast:
					if self.random_contrast_per_ch:
						for k in range(nb_channels):
							contrast_rand= np.random.uniform(low=self.contrast_min, high=self.contrast_max)
							contrasts.append(contrast_rand)
					else:
						contrast_rand= np.random.uniform(low=self.contrast_min, high=self.contrast_max)
						contrasts= [contrast_rand]*nb_channels
				else:
					contrasts= [self.contrast]*nb_channels

			# - Apply zscale stretch
			logger.debug("Applying zscale transform to batch %d with contrasts %s " % (i+1, str(contrasts)))

			batch.images[i] = self.__get_zscale_image(image, contrasts)
			if batch.images[i] is None:
				raise Exception("ZScale augmented image at batch %d is None!" % (i+1))

		return batch
	
	
	def __get_zscale_image(self, data, contrasts=[]):
		""" Apply z-scale transform to single image (W,H,Nch) """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		cond= np.logical_and(data!=0, np.isfinite(data))

		# - Check contrast dim vs nchans
		nchans= data.shape[-1]
	
		if len(contrasts)<nchans:
			logger.error("Invalid contrasts given (contrast list size=%d < nchans=%d)" % (len(self.contrasts), nchans))
			return None
		
		# - Transform each channel
		data_stretched= np.copy(data)

		for i in range(data.shape[-1]):
			data_ch= data_stretched[:,:,i]
			transform= ZScaleInterval(contrast=contrasts[i]) # able to handle NANs
			data_transf= transform(data_ch)
			data_stretched[:,:,i]= data_transf

		# - Scale data
		data_stretched[~cond]= 0 # Restore 0 and nans set in original data

		return data_stretched
		



class PercentileThrAugmenter(iaa.meta.Augmenter):
	""" Sigma threshold as augmentation step """
	
	def __init__(self, 
		percentile=50, 
		random_percentile=False, 
		random_percentile_per_ch=False, percentile_min=50, percentile_max=60, 
		seed=None, name=None, random_state="deprecated", deterministic="deprecated"
	):
		""" Build class """

		# - Set parent class parameters
		super(PercentileThrAugmenter, self).__init__(
			seed=seed, name=name,
			random_state=random_state, deterministic=deterministic
		)

		self.percentile= percentile
		self.random_percentile= random_percentile
		self.random_percentile_per_ch= random_percentile_per_ch
		self.percentile_min= percentile_min
		self.percentile_max= percentile_max
		self.seed= seed

	def get_parameters(self):
		""" Get class parameters """
		return [self.percentile, self.random_percentile, self.random_percentile_per_ch, self.percentile_min, self.percentile_max]

	def _augment_batch_(self, batch, random_state, parents, hooks):
		""" Augment batch of images """
	
		# - Check input batch
		if batch.images is None:
			return batch

		images = batch.images
		nb_images = len(images)
		percentiles= []

		# - Set random seed if given
		if self.seed is not None:
			np.random.seed(self.seed)

		# - Loop over image batch
		for i in range(nb_images):
			image= images[i]
			nb_channels = image.shape[2]

			# - Set percentiles (fixed or random)
			if not percentiles:
				if self.random_percentile:
					if self.random_percentile_per_ch:
						for k in range(nb_channels):
							percentile_rand= np.random.uniform(low=self.percentile_min, high=self.percentile_max)
							percentiles.append(percentile_rand)
					else:
						percentile_rand= np.random.uniform(low=self.percentile_min, high=self.percentile_max)
						percentiles= [percentile_rand]*nb_channels
				else:
					percentiles= [self.percentile]*nb_channels

			# - Apply percentile filtering
			logger.debug("Applying percentile thresholding to batch %d with contrasts %s " % (i+1, str(percentiles)))

			batch.images[i] = self.__get_percentile_thresholded_image(image, percentiles)
			if batch.images[i] is None:
				raise Exception("Percentile-thresholded augmented image at batch %d is None!" % (i+1))

		return batch
	
	
	def __get_percentile_thresholded_image(self, data, percentiles=[]):
		""" Apply percentile thresholding to single image (W,H,Nch) """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		cond= np.logical_and(data!=0, np.isfinite(data))

		# - Check percentiles dim vs nchans
		nchans= data.shape[-1]
	
		if len(percentiles)<nchans:
			logger.error("Invalid percentiles given (percentile list size=%d < nchans=%d)" % (len(self.percentiles), nchans))
			return None
		
		# - Threshold each channel
		data_thresholded= np.copy(data)

		for i in range(data.shape[-1]):
			percentile= percentiles[i]
			data_ch= data_thresholded[:,:,i]
			cond_ch= np.logical_and(data_ch!=0, np.isfinite(data_ch))
			data_ch_1d= data_ch[cond_ch]
			
			p= np.percentile(data, percentile)

			data_ch[data_ch<p]= 0
			data_ch[~cond_ch]= 0
	
			data_thresholded[:,:,i]= data_ch

		# - Scale data
		data_thresholded[~cond]= 0 # Restore 0 and nans set in original data

		return data_thresholded



class SigmoidStretchAugmenter(iaa.meta.Augmenter):
	""" Apply sigmoid contrast adjustment as augmentation step """
	
	def __init__(self, 
		cutoff=0.5, 
		gain=10,
		random_gain=False, 
		random_gain_per_ch=False, gain_min=10, gain_max=20, 
		seed=None, name=None, random_state="deprecated", deterministic="deprecated"
	):
		""" Build class """

		# - Set parent class parameters
		super(SigmoidStretchAugmenter, self).__init__(
			seed=seed, name=name,
			random_state=random_state, deterministic=deterministic
		)

		self.cutoff= cutoff
		self.gain= gain
		self.random_gain= random_gain
		self.random_gain_per_ch= random_gain_per_ch
		self.gain_min= gain_min
		self.gain_max= gain_max
		self.seed= seed

	def get_parameters(self):
		""" Get class parameters """
		return [self.cutoff, self.gain, self.random_gain, self.random_gain_per_ch, self.gain_min, self.gain_max]

	def _augment_batch_(self, batch, random_state, parents, hooks):
		""" Augment batch of images """
	
		# - Check input batch
		if batch.images is None:
			return batch

		images = batch.images
		nb_images = len(images)
		gains= []

		# - Set random seed if given
		if self.seed is not None:
			np.random.seed(self.seed)

		# - Loop over image batch
		for i in range(nb_images):
			image= images[i]
			nb_channels = image.shape[2]

			# - Set gains (fixed or random)
			if not gains:
				if self.random_gain:
					if self.random_gain_per_ch:
						for k in range(nb_channels):
							gain_rand= np.random.uniform(low=self.gain_min, high=self.gain_max)
							gains.append(gain_rand)
					else:
						gain_rand= np.random.uniform(low=self.gain_min, high=self.gain_max)
						gains= [gain_rand]*nb_channels
				else:
					gains= [self.gain]*nb_channels

			# - Apply sigmoid contrast transform
			logger.debug("Applying sigmoid stretch to batch %d with gain %s " % (i+1, str(gains)))

			batch.images[i] = self.__get_transformed_image(image, gains)
			if batch.images[i] is None:
				raise Exception("Sigmoid stretch augmented image at batch %d is None!" % (i+1))

		return batch
	
	
	def __get_transformed_image(self, data, gains=[]):
		""" Apply sigmoid contrast stretch to single image (W,H,Nch) """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		cond= np.logical_and(data!=0, np.isfinite(data))

		# - Check gains dim vs nchans
		nchans= data.shape[-1]
	
		if len(gains)<nchans:
			logger.error("Invalid gains given (gain list size=%d < nchans=%d)" % (len(self.gains), nchans))
			return None
		
		# - Threshold each channel
		data_transformed= np.copy(data)

		for i in range(data.shape[-1]):
			data_ch= data[:,:,i]
			if data_ch.min()<0:
				data_norm= rescale_intensity(data_ch, out_range=(0.,1.))
				data_ch= data_norm
			data_transformed[:,:,i]= adjust_sigmoid(data_ch, cutoff=self.cutoff, gain=gains[i], inv=False)
	
		# - Scale data
		data_transformed[~cond]= 0 # Restore 0 and nans set in original data

		return data_transformed



class RandomCropResizeAugmenter(iaa.meta.Augmenter):
	""" Apply random crop and resize to image as augmentation step """
	
	def __init__(self, 
		crop_fract_min=0.7, crop_fract_max=1.0, 
		seed=None, name=None, random_state="deprecated", deterministic="deprecated"
	):
		""" Build class """

		# - Set parent class parameters
		super(RandomCropResizeAugmenter, self).__init__(
			seed=seed, name=name,
			random_state=random_state, deterministic=deterministic
		)

		# - Set class parameters 
		if crop_fract_min<=0 or crop_fract_min>=crop_fract_max:
			raise Exception("Expected crop_fract_min to be >0 and <crop_frac_max, got %f!" % (crop_fract_min))
		if crop_fract_max>1 or crop_fract_max<=crop_fract_min:
			raise Exception("Expected crop_fract_max to be <=1 and >crop_frac_min, got %f!" % (crop_fract_max))

		self.crop_fract_min= crop_fract_min
		self.crop_fract_max= crop_fract_max
		self.seed= seed

	def get_parameters(self):
		""" Get class parameters """
		return [self.crop_fract_min, self.crop_fract_max]

	def _augment_batch_(self, batch, random_state, parents, hooks):
		""" Augment batch of images """
	
		# - Check input batch
		if batch.images is None:
			return batch

		images = batch.images
		nb_images = len(images)
		contrasts= []

		# - Set random seed if given
		if self.seed is not None:
			np.random.seed(self.seed)

		# - Loop over image batch
		for i in range(nb_images):
			image= images[i]
			nb_channels = image.shape[2]
			ny= image.shape[0]
			nx= image.shape[1]

			# - Set crop size
			crop_fract_rand= np.random.uniform(low=self.crop_fract_min, high=self.crop_fract_max)
			crop_x= math.ceil(crop_fract_rand*nx)
			crop_y= math.ceil(crop_fract_rand*ny)
			
			# - Define augmentation
			crop_aug= iaa.KeepSizeByResize(
				iaa.CropToFixedSize(crop_y, crop_x, position='uniform')
			)

			# - Crop & resize
			logger.debug("Applying crop to batch %d with crop_fract %s " % (i+1, str(crop_fract_rand)))

			batch.images[i] = crop_aug.augment_image(image)
			if batch.images[i] is None:
				raise Exception("Crop augmented image at batch %d is None!" % (i+1))

		return batch
	

##############################
##   Augmenter
##############################
class Augmenter(object):
	""" Perform image augmentation according to given model """

	def __init__(self, augmenter_choice="v1", augmenter=None, **kwparams):
		""" Create a data pre-processor object """

		# - Set parameters
		if augmenter is None:
			self.__set_augmenters(augmenter_choice)
		else:
			self.augmenter= augmenter
			
		# - Define safe augmenters for masks
		MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes", "Fliplr", "Flipud", "CropAndPad", "Affine", "PiecewiseAffine"]
		

	def hook(images, augmenter, parents, default):
		"""Determines which augmenters to apply to masks."""
		# Augmenters that are safe to apply to masks
		# Some, such as Affine, have settings that make them unsafe, so always
		# test your augmentation on masks        
		return augmenter.__class__.__name__ in MASK_AUGMENTERS	

	######################################
	##     DEFINE PREDEFINED AUGMENTERS
	######################################
	def __set_augmenters(self, choice='v1'):
		""" Define and set augmenters """

		# - Define augmenter v1 (same of caesar-mrcnn tf1 implementation)
		naugmenters_applied= 2
		augmenter_v1= iaa.SomeOf((0, naugmenters_applied),
			[
  			iaa.Fliplr(1.0),
  	  	iaa.Flipud(1.0),
  	  	iaa.Affine(rotate=(-90, 90), mode='constant', cval=0.0),
				iaa.Affine(translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)}, mode='constant', cval=0.0)
			],
			random_order=True
		)
		
		# - Define augmenter v2
		augmenter_v2= iaa.Sequential(
			[
				iaa.OneOf([iaa.Fliplr(1.0), iaa.Flipud(1.0), iaa.Noop()]),
  			iaa.Affine(rotate=(-90, 90), mode='constant', cval=0.0),
				#iaa.Affine(translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)}, mode='constant', cval=0.0)
			]
		)
		
		# - Define augmenter v3
		zscaleStretch_aug= ZScaleAugmenter(contrast=0.25, random_contrast=True, random_contrast_per_ch=False, contrast_min=0.1, contrast_max=0.5)
		sigmoidStretch_aug= SigmoidStretchAugmenter(cutoff=0.5, gain=10, random_gain=True, random_gain_per_ch=False, gain_min=10, gain_max=30)
		percThr_aug= PercentileThrAugmenter(percentile=50, random_percentile=True, random_percentile_per_ch=False, percentile_min=40, percentile_max=60)
		
		augmenter_v3= iaa.Sequential(
			[
				zscaleStretch_aug,	
				iaa.OneOf([iaa.Fliplr(1.0), iaa.Flipud(1.0), iaa.Noop()]),
  			iaa.Affine(rotate=(-90, 90), mode='constant', cval=0.0)
  			#iaa.Affine(translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)}, mode='constant', cval=0.0)
			]
		)

		
		# - Set augmenter chosen
		if choice=='v1':
			self.augmenter= augmenter_v1
		elif choice=='v2':
			self.augmenter= augmenter_v2
		elif choice=='v3':
			self.augmenter= augmenter_v3
		else:
			logger.warn("Unknown choice (%s), setting v1 augmenter..." % (choice))
			self.augmenter= augmenter_v1

        
		
	def __call__(self, data, masks):
		""" Apply transformation and return transformed data """
			
		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None
			
		# - Check masks
		if not masks:
			logger.error("Input mask list is empty!")
			return None

		# - Make augmenters deterministic to apply similarly to images and masks
		augmenter_det = self.augmenter.to_deterministic()

		# - Augment data cube
		try:
			data_aug= augmenter_det.augment_image(data)
		except Exception as e:
			logger.error("Failed to augment data (err=%s)!" % str(e))
			return None
			
		# - Augment masks
		#   NB: Change mask to np.uint8 because imgaug doesn't support np.bool
		try:
			masks_aug= augmenter_det.augment_images([mask.astype(np.uint8) for mask in masks], hooks=imgaug.HooksImages(activator=self.hook))
		except Exception as e:
			logger.error("Failed to augment mask (err=%s)!" % str(e))
			return None

		return data_aug, masks_aug
		
