#!/usr/bin/env python

from __future__ import print_function

##################################################
###          MODULE IMPORT
##################################################
## STANDARD MODULES
import os
import sys
import json
import time
import argparse
import datetime
import random
import numpy as np

## TENSORFLOW MODULES
import tensorflow as tf

## USER MODULES
from mrcnn import logger
from mrcnn.config import CONFIG
from mrcnn.dataset import Dataset
from mrcnn.preprocessing import DataPreprocessor
from mrcnn.preprocessing import BkgSubtractor, SigmaClipper, SigmaClipShifter, Scaler, LogStretcher
from mrcnn.preprocessing import Resizer, MinMaxNormalizer, AbsMinMaxNormalizer, MaxScaler, AbsMaxScaler, ChanMaxScaler
from mrcnn.preprocessing import Shifter, Standardizer, ChanDivider, MaskShrinker, BorderMasker
from mrcnn.preprocessing import ChanResizer, ZScaleTransformer, Chan3Trasformer
from mrcnn.augmentation import Augmenter

#===========================
#==   IMPORT MPI
#===========================
try:
	from mpi4py import MPI as MPI
	comm= MPI.COMM_WORLD
	nproc= comm.Get_size()
	procId= comm.Get_rank()
except Exception as e:
	logger.warn("Failed to import mpi4py module (err=%s), cannot run in parallel ..." % str(e))
	MPI= None
	comm= None
	nproc= 1
	procId= 0


logger.info("Tensorflow executing in eager mode? %d" % (tf.executing_eagerly()))

############################################################
#        PARSE/VALIDATE ARGS
############################################################

def parse_args():
	""" Parse command line arguments """  
  
	# - Parse command line arguments
	parser = argparse.ArgumentParser(description='Mask R-CNN options')

	parser.add_argument("command", metavar="<command>", help="'train' or 'test'")

	# - DATA PRE-PROCESSING OPTIONS
	parser.add_argument('--imgsize', dest='imgsize', required=False, type=int, default=256, help='Size in pixel used to resize input image (default=256)')
	
	parser.add_argument('--normalize_minmax', dest='normalize_minmax', action='store_true',help='Normalize each channel in range [0,1]')	
	parser.set_defaults(normalize_minmax=False)
	parser.add_argument('-norm_min', '--norm_min', dest='norm_min', required=False, type=float, default=0., action='store',help='Normalization min value (default=0)')
	parser.add_argument('-norm_max', '--norm_max', dest='norm_max', required=False, type=float, default=1., action='store',help='Normalization max value (default=0)')
	
	parser.add_argument('--subtract_bkg', dest='subtract_bkg', action='store_true',help='Subtract bkg from ref channel image')	
	parser.set_defaults(subtract_bkg=False)
	parser.add_argument('-sigma_bkg', '--sigma_bkg', dest='sigma_bkg', required=False, type=float, default=3, action='store',help='Sigma clip to be used in bkg calculation (default=3)')
	parser.add_argument('--use_box_mask_in_bkg', dest='use_box_mask_in_bkg', action='store_true',help='Compute bkg value in borders left from box mask')	
	parser.set_defaults(use_box_mask_in_bkg=False)	
	parser.add_argument('-bkg_box_mask_fract', '--bkg_box_mask_fract', dest='bkg_box_mask_fract', required=False, type=float, default=0.7, action='store',help='Size of mask box dimensions with respect to image size used in bkg calculation (default=0.7)')
	parser.add_argument('-bkg_chid', '--bkg_chid', dest='bkg_chid', required=False, type=int, default=-1, action='store',help='Channel to subtract background (-1=all) (default=-1)')

	parser.add_argument('--clip_shift_data', dest='clip_shift_data', action='store_true',help='Do sigma clipp shifting')	
	parser.set_defaults(clip_shift_data=False)
	parser.add_argument('-sigma_clip', '--sigma_clip', dest='sigma_clip', required=False, type=float, default=1, action='store',help='Sigma threshold to be used for clip & shifting pixels (default=1)')
	parser.add_argument('--clip_data', dest='clip_data', action='store_true',help='Do sigma clipping')	
	parser.set_defaults(clip_data=False)
	parser.add_argument('-sigma_clip_low', '--sigma_clip_low', dest='sigma_clip_low', required=False, type=float, default=10, action='store',help='Lower sigma threshold to be used for clipping pixels below (mean-sigma_low*stddev) (default=10)')
	parser.add_argument('-sigma_clip_up', '--sigma_clip_up', dest='sigma_clip_up', required=False, type=float, default=10, action='store',help='Upper sigma threshold to be used for clipping pixels above (mean+sigma_up*stddev) (default=10)')	
	parser.add_argument('-clip_chid', '--clip_chid', dest='clip_chid', required=False, type=int, default=-1, action='store',help='Channel to clip data (-1=all) (default=-1)')
	
	parser.add_argument('--zscale_stretch', dest='zscale_stretch', action='store_true',help='Do zscale transform')	
	parser.set_defaults(zscale_stretch=False)
	parser.add_argument('--zscale_contrasts', dest='zscale_contrasts', required=False, type=str, default='0.25,0.25,0.25',help='zscale contrasts applied to all channels') 
	
	parser.add_argument('--chan3_preproc', dest='chan3_preproc', action='store_true',help='Use the 3 channel pre-processor')	
	parser.set_defaults(chan3_preproc=False)
	parser.add_argument('-sigma_clip_baseline', '--sigma_clip_baseline', dest='sigma_clip_baseline', required=False, type=float, default=0, action='store',help='Lower sigma threshold to be used for clipping pixels below (mean-sigma_low*stddev) in first channel of 3-channel preprocessing (default=0)')
	parser.add_argument('-nchannels', '--nchannels', dest='nchannels', required=False, type=int, default=-1, action='store',help='Number of channels (if -1=take from data generator). If you modify channels in preprocessing you must set this (default=-1)')
	
	#parser.add_argument('--grayimg', dest='grayimg', action='store_true')	
	#parser.set_defaults(grayimg=False)
	#parser.add_argument('--no_uint8', dest='to_uint8', action='store_false')	
	#parser.set_defaults(to_uint8=True)
	#parser.add_argument('--no_zscale', dest='zscale', action='store_false')	
	#parser.set_defaults(zscale=True)
	#parser.add_argument('--zscale_contrasts', dest='zscale_contrasts', required=False, type=str, default='0.25,0.25,0.25',help='zscale contrasts applied to all channels') 
	#parser.add_argument('--biascontrast', dest='biascontrast', action='store_true')	
	#parser.set_defaults(biascontrast=False)
	#parser.add_argument('--bias', dest='bias', required=False, type=float, default=0.5,help='Bias value (default=0.5)') 
	#parser.add_argument('--contrast', dest='contrast', required=False, type=float, default=1.0,help='Contrast value (default=1)') 
	#parser.add_argument('--no_norm_img', dest='norm_img', action='store_false')	
	#parser.set_defaults(norm_img=True)	
	
	# - DATA AUGMENTATION OPTIONS
	parser.add_argument('--use_augmentation', dest='use_augmentation', action='store_true',help='Augment images')	
	parser.set_defaults(use_augmentation=False)
	parser.add_argument('-augmenter', '--augmenter', dest='augmenter', required=False, type=str, default='v1', action='store',help='Predefined augmenter to be used (default=v1)')

	# - MODEL OPTIONS
	parser.add_argument('--classdict', dest='classdict', required=False, type=str, default='{"sidelobe":1,"source":2,"galaxy":3}',help='Class id dictionary used when loading dataset') 
	parser.add_argument('--classdict_model', dest='classdict_model', required=False, type=str, default='',help='Class id dictionary used for the model (if empty, it is set equal to classdict)')
	parser.add_argument('--remap_classids', dest='remap_classids', action='store_true')	
	parser.set_defaults(remap_classids=False)
	parser.add_argument('--classid_remap_dict', dest='classid_remap_dict', required=False, type=str, default='',help='Dictionary used to remap detected classid to gt classid')
 
 	# - DATA LOADER
	#parser.add_argument('--dataloader',required=False,metavar="Data loader type",type=str,default='datalist',help='Train/cross-val data loader type {datalist,datalist_json,datadir_json}')
	parser.add_argument('--datalist', required=False, metavar="/path/to/dataset", help='Train/test data filelist containing a list of json files')
	#parser.add_argument('--datalist_train', required=False,metavar="/path/to/train_dataset", default=None, help='Train data filelist with format: filename_img,filename_mask,label or: filename_json')
	parser.add_argument('--datalist_val', required=False, metavar="/path/to/val_dataset", default=None, help='Validation data filelist containing a list of json files')
	#parser.add_argument('--datadir', required=False,metavar="/path/to/dataset",help='Train/test data top dir traversed to search json dataset files')
	#parser.add_argument('--validation_data_fract', dest='validation_data_fract', required=False, default=0.1, help='Fraction of input data used for cross-validation (default=0.1)')
	parser.add_argument('--maxnimgs', required=False, metavar="", type=int, default=-1, help="Max number of images to consider in dataset (-1=all) (default=-1)")
	
	
	# - TRAIN OPTIONS
	parser.add_argument('--weights', required=False, metavar="/path/to/weights.h5", help="Path to weights .h5 file")
	parser.add_argument('--ngpu', required=False,default=1,type=int,metavar="Number of GPUs",help='Number of GPUs')
	parser.add_argument('--nimg_per_gpu', required=False,default=1,type=int,metavar="Number of images per gpu",help='Number of images per gpu (default=1)')
	parser.add_argument('--nepochs', required=False,default=1,type=int,metavar="Number of training epochs",help='Number of training epochs (default=1)')
	parser.add_argument('--epoch_length', required=False,type=int,default=None,metavar="Number of data batches per epoch",help='Number of data batches per epoch. If None, assumed equal to the train sample size.')
	parser.add_argument('--nvalidation_steps', required=False,default=None,type=int,metavar="Number of validation steps per epoch",help='Number of validation steps per epoch. If None, assumed equal to the cross-validation sample size.')
	parser.add_argument('--rpn_anchor_scales', dest='rpn_anchor_scales', required=False, type=str, default='4,8,16,32,64',help='RPN anchor scales') 
	parser.add_argument('--max_gt_instances', dest='max_gt_instances', required=False, type=int, default=300,help='Max GT instances') 
	parser.add_argument('--backbone', dest='backbone', required=False, type=str, default='resnet101',help='Backbone network {resnet101,resnet50,custom} (default=resnet101)') 
	parser.add_argument('--backbone_strides', dest='backbone_strides', required=False, type=str, default='4,8,16,32,64',help='Backbone strides') 
	parser.add_argument('--rpn_nms_threshold', dest='rpn_nms_threshold', required=False, type=float, default=0.7,help='RPN Non-Maximum-Suppression threshold (default=0.7)') 
	parser.add_argument('--rpn_train_anchors_per_image', dest='rpn_train_anchors_per_image', required=False, type=int, default=512,help='Number of anchors per image to use for RPN training (default=512)')
	parser.add_argument('--train_rois_per_image', dest='train_rois_per_image', required=False, type=int, default=512,help='Number of ROIs per image to feed to classifier/mask heads (default=512)')
	parser.add_argument('--rpn_anchor_ratios', dest='rpn_anchor_ratios', required=False, type=str, default='0.5,1,2',help='RPN anchor ratios') 
	
	parser.add_argument('--rpn_class_loss_weight', dest='rpn_class_loss_weight', required=False, type=float, default='1',help='RPN classification loss weight') 
	parser.add_argument('--rpn_bbox_loss_weight', dest='rpn_bbox_loss_weight', required=False, type=float, default='1',help='RPN bounding box loss weight') 
	parser.add_argument('--mrcnn_class_loss_weight', dest='mrcnn_class_loss_weight', required=False, type=float, default='1',help='Classification loss weight') 
	parser.add_argument('--mrcnn_bbox_loss_weight', dest='mrcnn_bbox_loss_weight', required=False, type=float, default='1',help='Bounding box loss weight') 
	parser.add_argument('--mrcnn_mask_loss_weight', dest='mrcnn_mask_loss_weight', required=False, type=float, default='1',help='Mask loss weight') 
	
	parser.add_argument('--rpn_class_loss', dest='rpn_class_loss', action='store_true')
	parser.add_argument('--no_rpn_class_loss', dest='rpn_class_loss', action='store_false')
	parser.set_defaults(rpn_class_loss=True)

	parser.add_argument('--rpn_bbox_loss', dest='rpn_bbox_loss', action='store_true')
	parser.add_argument('--no_rpn_bbox_loss', dest='rpn_bbox_loss', action='store_false')
	parser.set_defaults(rpn_bbox_loss=True)
	
	parser.add_argument('--mrcnn_class_loss', dest='mrcnn_class_loss', action='store_true')
	parser.add_argument('--no_mrcnn_class_loss', dest='mrcnn_class_loss', action='store_false')
	parser.set_defaults(mrcnn_class_loss=True)

	parser.add_argument('--mrcnn_bbox_loss', dest='mrcnn_bbox_loss', action='store_true')
	parser.add_argument('--no_mrcnn_bbox_loss', dest='mrcnn_bbox_loss', action='store_false')
	parser.set_defaults(mrcnn_bbox_loss=True)

	parser.add_argument('--mrcnn_mask_loss', dest='mrcnn_mask_loss', action='store_true')
	parser.add_argument('--no_mrcnn_mask_loss', dest='mrcnn_mask_loss', action='store_false')
	parser.set_defaults(mrcnn_mask_loss=True)

	#parser.add_argument('--mask_loss_function', dest='mask_loss_function', required=False, type=str, default='binary_crossentropy', choices=['binary_crossentropy', 'dice_coef_loss'], help="Which loss function to use for mask loss. Accepted values are: binary_crossentropy and dice_coef_loss")

	parser.add_argument('--weight_classes', dest='weight_classes', action='store_true')	
	parser.set_defaults(weight_classes=False)
	
	parser.add_argument('--exclude_first_layer_weights', dest='exclude_first_layer_weights', action='store_true')	
	parser.set_defaults(exclude_first_layer_weights=False)

	#parser.add_argument('--no_augmentation', dest='use_augmentation', action='store_false')
	#parser.set_defaults(use_augmentation=True)

	# - TEST OPTIONS
	parser.add_argument('--scoreThr', required=False,default=0.7,type=float,metavar="Object detection score threshold to be used during test",help="Object detection score threshold to be used during test")
	parser.add_argument('--iouThr', required=False,default=0.6,type=float,metavar="IOU threshold used to match detected objects with true objects",help="IOU threshold used to match detected objects with true objects")

	parser.add_argument('--consider_sources_near_mixed_sidelobes', dest='consider_sources_near_mixed_sidelobes', action='store_true')
	parser.add_argument('--no_consider_sources_near_mixed_sidelobes', dest='consider_sources_near_mixed_sidelobes', action='store_false')
	parser.set_defaults(consider_sources_near_mixed_sidelobes=True)

	# - DETECT OPTIONS
	parser.add_argument('--image',required=False,metavar="Input image",type=str,help='Input image in FITS format to apply the model (used in detect task)')
	parser.add_argument('--xmin', dest='xmin', required=False, type=int, default=-1, help='Image min x to be read (read all if -1)') 
	parser.add_argument('--xmax', dest='xmax', required=False, type=int, default=-1, help='Image max x to be read (read all if -1)') 
	parser.add_argument('--ymin', dest='ymin', required=False, type=int, default=-1, help='Image min y to be read (read all if -1)') 
	parser.add_argument('--ymax', dest='ymax', required=False, type=int, default=-1, help='Image max y to be read (read all if -1)') 
	parser.add_argument('--detect_outfile',required=False,metavar="Output plot filename",type=str,default="",help='Output plot PNG filename (internally generated if left empty)')
	parser.add_argument('--detect_outfile_json',required=False,metavar="Output json filename with detected objects",type=str,default="",help='Output json filename with detected objects (internally generated if left empty)')

	# - PARALLEL PROCESSING OPTIONS
	parser.add_argument('--split_img_in_tiles', dest='split_img_in_tiles', action='store_true')	
	parser.set_defaults(split_img_in_tiles=False)
	parser.add_argument('--tile_xsize', dest='tile_xsize', required=False, type=int, default=512, help='Sub image size in pixel along x') 
	parser.add_argument('--tile_ysize', dest='tile_ysize', required=False, type=int, default=512, help='Sub image size in pixel along y') 
	parser.add_argument('--tile_xstep', dest='tile_xstep', required=False, type=float, default=1.0, help='Sub image step fraction along x (=1 means no overlap)') 
	parser.add_argument('--tile_ystep', dest='tile_ystep', required=False, type=float, default=1.0, help='Sub image step fraction along y (=1 means no overlap)') 

	# - RUN OPTIONS
	#parser.add_argument('--logs', required=False, default=DEFAULT_LOGS_DIR, metavar="/path/to/logs/", help='Logs and checkpoints directory (default=logs/)')
	#parser.add_argument('--nthreads', required=False, default=1, type=int, metavar="Number of worker threads", help="Number of worker threads")


	args = parser.parse_args()

	return args


def validate_args(args):
	""" Validate arguments """
	
	# - Check commands
	if args.command != "train" and args.command != "test" and args.command != "detect":
		logger.error("Unknow command (%s) given, only train/test/detect supported!" % args.command)
		return -1

	# - Check data loaders
	if args.command == "train" or args.command == "test":
		has_datalist= (args.datalist and args.datalist!="")
		if not has_datalist:
			logger.error("Argument --datalist is required for train/test tasks!")
			return -1
	
	# - Check image arg
	if args.command=='detect':
		has_image= (args.image and args.image!="")
		image_exists= os.path.isfile(args.image)
		valid_extension= args.image.endswith('.fits')
		if not has_image:
			logger.error("Argument --image is required for detect task!")
			return -1
		if not image_exists:
			logger.error("Image argument must be an existing image on filesystem!")
			return -1
		if not valid_extension:
			logger.error("Image must have .fits extension!")
			return -1

	# - Check maxnimgs
	if args.maxnimgs==0 or (args.maxnimgs<0 and args.maxnimgs!=-1):
		logger.error("Invalid maxnimgs given (hint: give -1 or >0)!")
		return -1

	# - Check weight file exists
	# ...

	# - Check remap id
	if args.remap_classids:
		if args.classid_remap_dict=="":
			logger.error("Classid remap dictionary is empty (you need to provide one if you give the option --remap_classids)!")
			return -1

	return 0

############################################################
#       MAIN
############################################################
def main():
	"""Main function"""

	#===========================
	#==   PARSE ARGS
	#===========================
	if procId==0:
		logger.info("[PROC %d] Parsing script args ..." % procId)
	try:
		args= parse_args()
	except Exception as ex:
		logger.error("[PROC %d] Failed to get and parse options (err=%s)" % (procId, str(ex)))
		return 1

	#===========================
	#==   VALIDATE ARGS
	#===========================
	if procId==0:
		logger.info("[PROC %d] Validating script args ..." % procId)
	if validate_args(args)<0:
		logger.error("[PROC %d] Argument validation failed, exit ..." % procId)
		return 1

	if procId==0:
		print("Weights: ", args.weights)
		print("Datalist: ", args.datalist)
		print("Logs: ", args.logs)
		print("nEpochs: ", args.nepochs)
		print("epoch_length: ",args.epoch_length)
		print("nvalidation_steps: ",args.nvalidation_steps)
		print("ngpu: ",args.ngpu)
		print("nimg_per_gpu: ",args.nimg_per_gpu)
		print("scoreThr: ",args.scoreThr)
		print("classdict: ",args.classdict)

	#===========================
	#==   SET PARAMETERS
	#===========================
	# - Set data pre-processing options
	zscale_contrasts= [float(x) for x in args.zscale_contrasts.split(',')]
	if args.chan3_preproc and args.nchannels!=3:
		logger.error("You selected chan3_preproc pre-processing options, you must set nchannels options to 3!")
		return 1
	
	# - Set model options
	weights_path = args.weights

	rpn_anchor_scales= tuple([int(x.strip()) for x in args.rpn_anchor_scales.split(',')])
	backbone_strides= [int(x.strip()) for x in args.backbone_strides.split(',')]
	rpn_anchor_ratios= [float(x.strip()) for x in args.rpn_anchor_ratios.split(',')]

	train_from_scratch= False
	if not weights_path or weights_path=='':
		train_from_scratch= True

	exclude_first_layer_weights= args.exclude_first_layer_weights

	# - Set class label-target dictionary for dataset
	try:
		class_dict= json.loads(args.classdict)
	except:
		logger.error("[PROC %d] Failed to convert class dict string to dict!" % (procId))
		return -1	

	# - Set class label-target dictionary for model
	#   NB: Add background class if not added
	class_dict_model= class_dict
	if args.classdict_model!="":
		try:
			class_dict_model= json.loads(args.classdict_model)
		except:
			logger.error("[PROC %d] Failed to convert class dict model string to dict!" % (procId))
			return -1		
			
	if 'background' not in class_dict_model:
		logger.info("[PROC %d] background class not present in dictionary, adding it and set it to target 0 ..." % (procId)) 
		class_dict_model['background']= 0	

	
	# - Compute class names
	nclasses= len(class_dict)
	nclasses_model= len(class_dict_model)
	class_names= [item for item in class_dict]
	class_names_model= [item for item in class_dict_model]
	
	if procId==0:
		logger.info("[PROC %d] Assuming #%d classes in dataset from given class dictionary ..." % (procId, nclasses))
		print("CLASS_NAMES (DATASET)")
		print(class_names)
	
		logger.info("[PROC %d] Assuming #%d classes in model from given class dictionary ..." % (procId, nclasses_model))
		print("CLASS_NAMES (MODEL)")
		print(class_names_model)

	#loss_weight_dict= {}
	#loss_weight_dict['rpn_class_loss']= args.rpn_class_loss_weight
	#loss_weight_dict['rpn_bbox_loss']= args.rpn_bbox_loss_weight
	#loss_weight_dict['mrcnn_class_loss']= args.mrcnn_class_loss_weight
	#loss_weight_dict['mrcnn_bbox_loss']= args.mrcnn_bbox_loss_weight
	#loss_weight_dict['mrcnn_mask_loss']= args.mrcnn_mask_loss_weight
	
	loss_weights= [args.rpn_class_loss_weight, args.rpn_bbox_loss_weight, args.mrcnn_class_loss_weight, args.mrcnn_bbox_loss_weight, args.mrcnn_mask_loss_weight]

	#use_loss_dict= {}
	#use_loss_dict['rpn_class_loss']= True
	#use_loss_dict['rpn_bbox_loss']= True
	#use_loss_dict['mrcnn_class_loss']= True
	#use_loss_dict['mrcnn_bbox_loss']= True
	#use_loss_dict['mrcnn_mask_loss']= True
 
	#if not args.rpn_class_loss:
	#	use_loss_dict['rpn_class_loss']= False
	#if not args.rpn_bbox_loss:
	#	use_loss_dict['rpn_bbox_loss']= False
	#if not args.mrcnn_class_loss:
	#	use_loss_dict['mrcnn_class_loss']= False
	#if not args.mrcnn_bbox_loss:
	#	use_loss_dict['mrcnn_bbox_loss']= False
	#if not args.mrcnn_mask_loss:
	#	use_loss_dict['mrcnn_mask_loss']= False

	#mask_loss_function = args.mask_loss_function

	consider_sources_near_mixed_sidelobes = args.consider_sources_near_mixed_sidelobes


	#==============================
	#==   DEFINE PRE-PROCESSOR
	#==============================
	preprocess_stages= []

	if subtract_bkg:
		preprocess_stages.append(BkgSubtractor(sigma=sigma_bkg, use_mask_box=use_box_mask_in_bkg, mask_fract=bkg_box_mask_fract, chid=bkg_chid))

	if clip_shift_data:
		preprocess_stages.append(SigmaClipShifter(sigma=sigma_clip, chid=clip_chid))

	if clip_data:
		preprocess_stages.append(SigmaClipper(sigma_low=sigma_clip_low, sigma_up=sigma_clip_up, chid=clip_chid))

	if zscale_stretch:
		preprocess_stages.append(ZScaleTransformer(contrasts=zscale_contrasts))

	if chan3_preproc:
		preprocess_stages.append( Chan3Trasformer(sigma_clip_baseline=sigma_clip_baseline, sigma_clip_low=sigma_clip_low, sigma_clip_up=sigma_clip_up, zscale_contrast=zscale_contrasts[0]) )

	if resize:
		preprocess_stages.append(Resizer(resize_size=resize_size, upscale=upscale, downscale_with_antialiasing=downscale_with_antialiasing, set_pad_val_to_min=set_pad_val_to_min))

	if normalize_minmax:
		preprocess_stages.append(MinMaxNormalizer(norm_min=norm_min, norm_max=norm_max))

	logger.info("[PROC %d] Data pre-processing steps: %s" % (procId, str(preprocess_stages)))
	#print(preprocess_stages)

	dp= DataPreprocessor(preprocess_stages)

	# - Creating data pre-processor for validation/test data
	#logger.info("Creating data pre-processor for validation/test data ...")
	#dp_val= DataPreprocessor(preprocess_stages_val)
	
	#==============================
	#==   DEFINE AUGMENTER
	#==============================
	augmenter= None
	if args.use_augmentation:
		logger.info("[PROC %d] Setting augmenter %s ..." % (procId, args.augmenter))
		augmenter= Augmenter(augmenter_choice=args.augmenter)
		
	#==============================
	#==   LOAD DATASETS
	#==============================
	# - Loading train/test dataset
	logger.info("[PROC %d] Creating train/test dataset ..." % (procId))
	dataset= Dataset(
		preprocess_transform=dp,
		augmentation=augmenter,
		**CONFIG
	)
	dataset.set_class_dict(class_dict)
	
	logger.info("[PROC %d] Loading dataset from file %s ..." % (procId, args.datalist))
		
	dataset.load_data_from_json_list(args.datalist, args.maxnimgs)
	
	# - Loading validation dataset (if enabled)
	dataset_cv= None
	if args.datalist_val!="":
		logger.info("[PROC %d] Creating validation dataset ..." % (procId))
	
		dataset_cv= Dataset(
			preprocess_transform=dp,
			augmentation=None,
			**CONFIG
		)
		dataset_cv.set_class_dict(class_dict)
	
		logger.info("[PROC %d] Loading dataset from file %s ..." % (procId, args.datalist))
		dataset_cv.load_data_from_json_list(args.datalist_val, args.maxnimgs)
	
	#if args.command == "train":
	#	if procId==0:
	#		logger.info("[PROC %d] Creating/setting & loading train/val datasets ..." % procId)
		#datasets= create_train_val_datasets(args)
		#if len(datasets)!=2:
		#	logger.error("[PROC %d] Failed to create train/val datasets!" % procId)
		#	return 1

	#elif args.command == "test":
	#	if procId==0:
	#		logger.info("[PROC %d] Creating/setting & loading test dataset ..." % procId)
	#	dataset= create_test_dataset(args)
	#	if dataset is None:
	#		logger.error("[PROC %d] Failed to create test dataset!" % procId)
	#		return 1

	# - Updating steps per epoch
	#if args.command == "train":
	#	nentries_train= datasets[0].loaded_imgs
	#	nentries_val= datasets[1].loaded_imgs
	#	epoch_step_info_given= ( (args.epoch_length is not None) and (args.epoch_length>0) )
	#	val_step_info_given= ( (args.nvalidation_steps is not None) and (args.nvalidation_steps>0) )

	#	if epoch_step_info_given and val_step_info_given:
	#		steps_per_epoch= ((args.epoch_length - args.nvalidation_steps) // (args.nimg_per_gpu*args.ngpu))
	#		validation_steps_per_epoch= max(1, args.nvalidation_steps // (args.nimg_per_gpu*args.ngpu))

	#	else:
	#		steps_per_epoch= (nentries_train // (args.nimg_per_gpu*args.ngpu))
	#		validation_steps_per_epoch= max(1, nentries_val // (args.nimg_per_gpu*args.ngpu))

	#	if procId==0:
	#		logger.info("[PROC %d] Train/validation steps per epoch= %d/%d" % (procId,steps_per_epoch,validation_steps_per_epoch))

	#elif args.command == "test":
	#	steps_per_epoch= 1
	#	validation_steps_per_epoch= 1

	
	#===========================
	#==   CONFIG
	#===========================
	# - Override some other options
	logger.info("Setting config options ...")
	
	CONFIG.update(
		{
			'class_dict': class_dict_model,
			'num_classes': nclasses_model,
    	'epochs': args.nepochs,
		},
	)
	CONFIG.update({'meta_shape': (1 + 3 + 3 + 4 + 1 + CONFIG['num_classes']),})
	
	CONFIG['images_per_gpu']= args.nimg_per_gpu
	CONFIG['batch_size']= args.ngpu*args.nimg_per_gpu
	CONFIG['rpn_anchor_scales']= rpn_anchor_scales
	CONFIG['max_gt_instances']= args.max_gt_instances
	CONFIG['backbone']= args.backbone
	CONFIG['backbone_strides']= backbone_strides
	CONFIG['rpn_nms_threshold']= args.rpn_nms_threshold
	CONFIG['rpn_train_anchors_per_image']= args.rpn_train_anchors_per_image
	CONFIG['train_rois_per_image']= args.train_rois_per_image
	CONFIG['rpn_anchor_ratios']= rpn_anchor_ratios
	CONFIG['loss_weights']= loss_weights
	CONFIG['image_min_dim']= args.imgsize
	CONFIG['image_max_dim']= args.imgsize
	CONFIG['image_shape']= ([args.imgsize, args.imgsize, 3])
	if args.grayimg:
		CONFIG['image_shape']= ([args.imgsize, args.imgsize, 1])

	if args.command == "train":
		CONFIG['training']= True
		CONFIG['gpu_num']= args.ngpu
		
	elif args.command == "test":
		CONFIG['training']= False
		CONFIG['gpu_num']= 1
		
	elif args.command == "detect":
		CONFIG['training']= False
		CONFIG['gpu_num']= 1
	
	# - Set addon options
	#config.IMG_PATH= args.image
	#config.IMG_XMIN= args.xmin
	#config.IMG_XMAX= args.xmax
	#config.IMG_YMIN= args.ymin
	#config.IMG_YMAX= args.ymax
	#config.ZSCALE_STRETCH= args.zscale
	#config.ZSCALE_CONTRASTS= [float(x) for x in args.zscale_contrasts.split(',')]
	#config.NORMALIZE_IMG= args.norm_img
	#config.IMG_TO_UINT8= args.to_uint8
	#config.IMG_TO_RGB= not args.grayimg
	#if args.grayimg:
	#	config.IMAGE_CHANNEL_COUNT= 1 # for gray-level images (see https://github.com/matterport/Mask_RCNN/wiki)
	#	config.MEAN_PIXEL= 0          # for gray-level images

	#config.BIAS_CONTRAST_STRETCH= args.biascontrast
	#config.IMG_BIAS= args.bias
	#config.IMG_CONTRAST= args.contrast

	#config.IOU_THR= args.iouThr
	#config.SCORE_THR= args.scoreThr

	#config.MPI= MPI
	#config.SPLIT_IMG_IN_TILES= args.split_img_in_tiles
	#config.TILE_XSIZE= args.tile_xsize
	#config.TILE_YSIZE= args.tile_ysize
	#config.TILE_XSTEP= args.tile_xstep
	#config.TILE_YSTEP= args.tile_ystep

	#config.OUTFILE= args.detect_outfile
	#config.OUTFILE_JSON= args.detect_outfile_json

	
	logger.info("Config options: %s" % (str(CONFIG)))

	#===========================
	#==   CREATE MODEL
	#===========================
	# - Create model
	#if args.command == "train":
	#	model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)
	#elif args.command == "test" or args.command == "detect":
	#	DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
	#	with tf.device(DEVICE):
	#		model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)

	#if procId==0:
	#	logger.info("[PROC %d] Printing the model ..." % procId)
	#	model.print_model()

	# - Load weights
	#if train_from_scratch:
	#	if procId==0:
	#		logger.info("[PROC %d] No weights given, training from scratch ..." % procId)
	#else:
	#	if procId==0:
	#		logger.info("[PROC %d] Loading weights from file %s (excluding first layer? %d) ..." % (procId, weights_path, exclude_first_layer_weights))

	#	if exclude_first_layer_weights:
	#		model.load_weights(weights_path, by_name=True, exclude='conv1')
	#	else:
	#		model.load_weights(weights_path, by_name=True)
	
	#===========================
	#==   TRAIN/TEST
	#===========================
	# - Train or evaluate
	#if args.command == "train":
	#	if train(args, model, config, datasets)<0:
	#		logger.error("[PROC %d] Failed to run train!" % procId)
	#		return 1
	#elif args.command == "test":
	#	if test(args, model, config, dataset)<0:
	#		logger.error("[PROC %d] Failed to run test!" % procId)
	#		return 1
	#elif args.command == "detect":
	#	if detect(args, model, config)<0:
	#		logger.error("[PROC %d] Failed to run detect!" % procId)
	#		return 1
	
	return 0


###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())