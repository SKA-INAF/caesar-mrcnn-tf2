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
import math
import datetime
import collections
import csv
import logging
from typing import List		# for type annotation
import numpy as np

# Import networkx module
import networkx

# Import image modules
import skimage.draw
import skimage.measure
from skimage.measure import find_contours
from sklearn.metrics import jaccard_score
from skimage import measure
from skimage.measure import regionprops
import cv2 as cv
import imutils

## ASTROPY MODULES
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

## Import graphics modules
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches, lines
from matplotlib.patches import Polygon

## Import regions module
import regions
from regions import PolygonPixelRegion, RectanglePixelRegion, PixCoord

## USER MODULES
from mrcnn import logger
from mrcnn.graph import Graph
from mrcnn import utils, addon_utils, visualize


# ========================
# ==    MODEL TESTER
# ========================
class ModelTester(object):
	""" Define model tester class """

	def __init__(self, model, config, dataset):
		""" Return model tester class """

		self.dataset= dataset
		self.model= model
		self.config= config

		# - Data options
		self.n_max_img= -1
		self.remap_classids= False
		self.classid_map= {}

		# - Process options
		self.score_thr= 0.7
		self.iou_thr= 0.6

		# - Results
		#self.n_classes= self.config.NUM_CLASSES
		self.n_classes= dataset.nclasses
		self.classification_matrix= np.zeros((self.n_classes,self.n_classes))
		self.classification_matrix_norm= np.zeros((self.n_classes,self.n_classes))
		self.purity= np.zeros((1,self.n_classes))
		self.nobjs_true= np.zeros((1,self.n_classes))
		self.nobjs_det= np.zeros((1,self.n_classes))
		self.nobjs_det_right= np.zeros((1,self.n_classes))
		self.detobj_scores= []
		self.detobj_ious= []
		self.detobj_scoreMean= 0
		self.detobj_scoreStdDev= 0
		self.detobj_iouMean= 0
		self.detobj_iouStdDev= 0
		self.detobj_gtinfo= []
		self.compute_mAP_metrics= False
		self.mAP = 0

		# - Output file
		self.completeness_dict_list= []
		self.reliability_dict_list= []	
		self.outfilename_completeness= "completeness.csv"
		self.outfilename_reliability= "reliability.csv"
		
		# - Draw options
		self.save_plots= True
		self.draw= True
		self.draw_shaded_masks= False
		self.draw_class_label_in_caption= False

	# ========================
	# ==     TEST
	# ========================
	def test(self):
		""" Test the model on input dataset """    
	
		# - Loop over dataset and inspect results
		nimg= 0
		logger.info("Processing up to %d images " % (self.n_max_img))

		for index, image_id in enumerate(self.dataset.image_ids):

			# - Check if stop inspection
			if self.n_max_img>0 and nimg>=self.n_max_img:
				logger.info("Max number of images to inspect reached, stop here.")
				break

			nimg+= 1

			# - Inspect results	for current image
			image_path = self.dataset.image_info[index]['path']
			image_path_base= os.path.basename(image_path)

			# - Initialize the analyzer
			analyzer= Analyzer(self.model, self.config, self.dataset)
			analyzer.score_thr= self.score_thr
			analyzer.iou_thr= self.iou_thr
			analyzer.remap_classids= self.remap_classids
			analyzer.classid_map= self.classid_map
			analyzer.save_plots= self.save_plots
			analyzer.draw= self.draw
			analyzer.draw_shaded_masks= self.draw_shaded_masks
			analyzer.draw_class_label_in_caption= self.draw_class_label_in_caption

			# - Inspecting results
			logger.info("Inspecting results for image %s ..." % image_path_base)
			status= analyzer.inspect_results(image_id, image_path)
			if status<0:
				logger.error("Failed to analyze results for image %s ..." % image_path_base)
				continue

			# - Update performances
			logger.info("Updating test performances using results for image %s ..." % image_path_base)
			self.update_performances(analyzer)

		# - Compute final results
		logger.info("Computing final performances ...")
		self.compute_performances()

		# - Save to file
		logger.info("Saving result data to file ...")
		self.save()

		return 0

	# =============================
	# ==     UPDATE PERFORMANCES
	# =============================
	def update_performances(self, analyzer):
		""" Update test performances using current sample """
		
		# - Retrieve perf data for sample image
		C_sample= analyzer.confusion_matrix
		nobjs_true_sample= analyzer.nobjs_true
		nobjs_det_sample= analyzer.nobjs_det
		nobjs_det_right_sample= analyzer.nobjs_det_right
		detobj_scores_sample= analyzer.detobj_scores
		detobj_ious_sample= analyzer.detobj_ious

		# - Sum perf data
		self.classification_matrix+= C_sample
		self.nobjs_true+= nobjs_true_sample
		self.nobjs_det+= nobjs_det_sample
		self.nobjs_det_right+= nobjs_det_right_sample
		self.detobj_scores+= detobj_scores_sample
		self.detobj_ious+= detobj_ious_sample

		# - Append data to completeness output dict
		image_path= analyzer.image_path
		image_tel= analyzer.image_metadata["telescope"]
		image_rms= analyzer.image_metadata["rms"]
		image_bkg= analyzer.image_metadata["bkg"]
		class_ids_gt= analyzer.class_ids_gt_merged
		objinfo_gt= analyzer.detobj_gtinfo
		scores_det= analyzer.detobj_scores
		ious_det= analyzer.detobj_ious
		class_ids_det= analyzer.detobj_classids
		class_names_det= analyzer.detobj_class_names
		is_gt_obj_detected= analyzer.is_gt_obj_detected
		
		if objinfo_gt:
			if len(class_ids_gt)!=len(objinfo_gt):
				logger.warn("classids_gt size is different from objinfo_gt size!")

			for i in range(len(objinfo_gt)):
				obj= objinfo_gt[i]
				is_flagged= obj['sidelobe-mixed']
				nislands= obj['nislands']
				at_border= obj['border']
				sname= obj['name']
				snr= obj['snr']
				maxBeamSize= obj['maxsize_beam']
				minBeamSize= obj['minsize_beam']
				aspectRatio= maxBeamSize/minBeamSize

				class_id= class_ids_gt[i]
				class_name= obj['class']
				detected= is_gt_obj_detected[i]
				class_id_det= class_ids_det[i]
				class_name_det= class_names_det[i]
				score_det= scores_det[i]
				iou_det= ious_det[i]
				
				d= collections.OrderedDict()
				d["img"]= image_path
				d["telescope"]= image_tel
				d["img_rms"]= image_rms
				d["img_bkg"]= image_bkg
				d["sname"]= sname
				d["class_id"]= class_id
				d["class_name"]= class_name
				d["class_id_det"]= class_id_det
				d["class_name_det"]= class_name_det
				d["detected"]= int(detected)
				d["score"]= score_det
				d["iou"]= iou_det
				d["snr"]= float(snr)
				d["maxBeamSize"]= float(maxBeamSize)
				d["aspectRatio"]= float(aspectRatio)
				d["border"]= int(at_border)
				self.completeness_dict_list.append(d)

		# - Fill reliability table dict
		objinfo_det= analyzer.det_obj_pars
		class_ids_det= analyzer.class_ids_final
		scores_det= analyzer.scores_final	
		is_det_obj_matching_to_gt_obj= analyzer.is_det_obj_matching_to_gt_obj
		matchobj_classids= analyzer.matchobj_classids
		matchobj_class_names= analyzer.matchobj_class_names
		matchobj_ious= analyzer.matchobj_ious


		if objinfo_det:
			if len(class_ids_det)!=len(objinfo_det):
				logger.warn("class_ids_det size is different from objinfo_det size!")

			for i in range(len(objinfo_det)):
				if not objinfo_det:
					logger.warn("Skipping this obj info det as empty dict (hint: possibly not filled as contour ops failed) ...")
				obj_det= objinfo_det[i]
				at_border= obj_det['border']
				sname= obj_det['name']
				snr= obj_det['snr']
				maxBeamSize= obj_det['maxsize_beam']
				minBeamSize= obj_det['minsize_beam']
				aspectRatio= -999
				if minBeamSize>0:
					aspectRatio= float(maxBeamSize)/float(minBeamSize)

				class_id_det= class_ids_det[i]
				class_name_det= obj_det['class']
				matching_gt= is_det_obj_matching_to_gt_obj[i]
				class_id= matchobj_classids[i]
				class_name= matchobj_class_names[i]
				iou_det= matchobj_ious[i]
				score_det= scores_det[i]

				d= collections.OrderedDict()
				d["img"]= image_path
				d["telescope"]= image_tel
				d["img_rms"]= image_rms
				d["img_bkg"]= image_bkg
				d["sname"]= sname
				d["class_id_det"]= class_id_det
				d["class_name_det"]= class_name_det
				d["class_id"]= class_id
				d["class_name"]= class_name
				d["matching_gt"]= int(matching_gt)
				d["score"]= float(score_det)
				d["iou"]= float(iou_det)
				d["snr"]= float(snr)
				d["maxBeamSize"]= float(maxBeamSize)
				d["aspectRatio"]= float(aspectRatio)
				d["border"]= int(at_border)
				self.reliability_dict_list.append(d)
			
		# - Write current data to file
		logger.info("Writing current completeness & reliability table data to file ...")
		self.save()

		return 0


	# =============================
	# ==     SAVE
	# =============================
	def save(self):
		""" Save data """	

		# - Save completeness table file
		if self.completeness_dict_list:
			logger.info("Saving completeness table file ...")
			parnames = self.completeness_dict_list[0].keys()
		
			with open(self.outfilename_completeness, 'w') as fp:
				fp.write("# ")
				dict_writer = csv.DictWriter(fp, parnames)
				dict_writer.writeheader()
				dict_writer.writerows(self.completeness_dict_list)

		# - Save reliability table file
		if self.reliability_dict_list:
			logger.info("Saving reliability table file ...")
			parnames = self.reliability_dict_list[0].keys()

			with open(self.outfilename_reliability, 'w') as fp:
				fp.write("# ")
				dict_writer = csv.DictWriter(fp, parnames)
				dict_writer.writeheader()
				dict_writer.writerows(self.reliability_dict_list)

	# =============================
	# ==     COMPUTE PERFORMANCES
	# =============================
	def compute_performances(self):
		""" Compute final performances """
		
		# - Normalize classification matrix
		for i in range(self.n_classes):
			norm= self.nobjs_true[0][i]
			if norm<=0:
				continue
			for j in range(self.n_classes):
				C= self.classification_matrix[i][j]
				C_norm= C/norm
				self.classification_matrix_norm[i][j]= C_norm

		# - Compute purity
		for j in range(self.n_classes):
			if self.nobjs_det[0][j]<=0:
				continue
			p= self.nobjs_det_right[0][j]/self.nobjs_det[0][j]
			self.purity[0][j]= p

		# - Compute score mean
		self.detobj_scoreMean= np.mean(self.detobj_scores)
		self.detobj_scoreStdDev= np.std(self.detobj_scores)
		self.detobj_iouMean= np.mean(self.detobj_ious)
		self.detobj_iouStdDev= np.std(self.detobj_ious)


		# - Print results
		print("== NOBJ TRUE ==")
		print(self.nobjs_true)

		print("== NOBJ DET ==")
		print(self.nobjs_det)

		print("== NOBJ DET CORRECTLY ==")
		print(self.nobjs_det_right)

		print("== CLASSIFICATION MATRIX ==")
		print(self.classification_matrix)

		print("== CLASSIFICATION MATRIX (NORM) ==")
		print(self.classification_matrix_norm)

		print("== PRECISION (or PURITY) ==")
		print(self.purity)

		print("== DET SCORES ==")
		print("scoreThr=%f, <score>=%f, sigma(score)=%f" % (self.score_thr,self.detobj_scoreMean,self.detobj_scoreStdDev))

		print("== DET IOUs ==")
		print("iouThr=%f, <iou>=%f, sigma(iou)=%f" % (self.iou_thr,self.detobj_iouMean,self.detobj_iouStdDev))


# ========================
# ==    ANALYZER
# ========================
class Analyzer(object):
	""" Define analyzer object """

	def __init__(self, model, config, dataset=None):
		""" Return an analyzer object """

		# - Model
		self.model= model
		self.r= None

		# - Config options
		self.config= config
		if dataset:
			self.n_classes= dataset.nclasses
		else:
			self.n_classes= self.config.NUM_CLASSES

		# - Data options
		self.dataset= dataset
		self.image= None
		self.image_header= None
		self.image_id= -1
		self.image_uuid= ''
		self.image_path= ''
		self.image_path_base= ''
		self.image_path_base_noext= ''
		self.image_xmin= 0
		self.image_ymin= 0
		
		# - Raw model data
		self.class_names= None
		self.masks= None
		self.boxes= None
		self.class_ids= None
		self.scores= None
		self.nobjects= 0

		# - Processed ground truth masks
		self.class_names_gt= None
		self.masks_gt_merged= []
		self.class_ids_gt_merged= []
		self.bboxes_gt= []
		self.captions_gt= []
		self.split_gtmasks= False
		self.sidelobes_mixed_or_near_gt_merged= []

		# - Processed detected masks
		self.masks_final= []
		self.class_ids_final= []
		self.class_names_final= []
		self.scores_final= []	
		self.bboxes= []
		self.captions= []
		self.remap_classids= False
		self.classid_map= {}
		self.split_masks= False
		self.merge_overlapped_masks= True
		self.select_best_overlapped_masks= True
		self.split_source_sidelobe= True
		#self.source_sidelobe_overlap_iou_thr= 0.1 # If they overlap less than thr, keep separate
		self.merge_overlap_iou_thr= 0.3 # overlapping objects with same class with IOU>thr are merged in a unique object
		self.det_obj_pars= []

		self.results= {}     # dictionary with detected objects
		self.obj_name_tag= ""
		self.obj_regions= [] # list of DS9 region objects

		# - Process options
		self.score_thr= 0.7
		self.iou_thr= 0.6

		# - Performances results
		self.detobj_scores= []
		self.detobj_ious= []
		self.detobj_gtinfo= []
		self.confusion_matrix= np.zeros((self.n_classes,self.n_classes))
		self.confusion_matrix_norm= np.zeros((self.n_classes,self.n_classes))	
		self.purity= np.zeros((1,self.n_classes))
		self.nobjs_true= np.zeros((1,self.n_classes))
		self.nobjs_det= np.zeros((1,self.n_classes))
		self.nobjs_det_right= np.zeros((1,self.n_classes))
		self.is_gt_obj_detected= []

		# - Draw options
		self.outfile= ""
		self.outfile_json= ""
		self.outfile_ds9= ""
		self.save_plots= False
		self.draw= True
		self.draw_shaded_masks= False
		self.draw_class_label_in_caption= False
		self.write_to_json= True
		self.write_to_ds9= True
		self.use_polygon_regions= True
		self.class_color_map= {
			'bkg': (0,0,0),# black
			'spurious': (1,0,0),# red
			'compact': (0,0,1),# blue
			'extended': (1,1,0),# green	
			'extended-multisland': (1,0.647,0),# orange
			'flagged': (0,0,0),# black
		}
		self.class_color_map_ds9= {
			'bkg': "black",
			'spurious': "red",
			'compact': "blue",
			'extended': "green",	
			'extended-multisland': "orange",
			'flagged': "magenta",
		}


	def set_image_path(self,path):
		""" Set image path """
		self.image_path= path
		self.image_path_base= os.path.basename(self.image_path)
		self.image_path_base_noext= os.path.splitext(self.image_path_base)[0]

	# =============================
	# ==     GET DATA FROM MODEL
	# =============================
	def get_data(self):
		""" Retrieve data from dataset & model """

		tstart= time.time()

		# - Throw error if dataset is not given
		if self.dataset is None:
			logger.error("No dataset given!")
			return -1
			
		###############
		##  TEST ######
		###############
		#image, gt_masks, gt_class_ids, gt_boxes, image_meta, original_image, original_masks_array, original_class_ids, original_bboxes = self.dataset[0]
		
		#print("IMAGE")
		#print(image.shape)
		#plt.imshow(image)
		#plt.show()
    
		#print("GT_MASKS")
		#print(gt_masks.shape)
		#for i in range(gt_masks.shape[-1]):
		#	print("GT_MASK %d" % (i+1))
		#	print(gt_masks[:,:,i].shape)
		#	plt.imshow(gt_masks[:,:,i])
		#	plt.show()
    	
		#print("GT_BOXES")
		#print(gt_boxes)	
		###############	

		# - Load image
		t1 = time.time()
		self.image, _, _= self.dataset.load_and_process_image(self.image_id, preprocess=True, resize=False)
		original_input_shape = self.image.shape
		image_processed, image_meta, window = self.dataset.process_image(self.image, self.image_id, preprocess=True, resize=True)
		self.image_path_base= os.path.basename(self.image_path)
		self.image_path_base_noext= os.path.splitext(self.image_path_base)[0]		
		self.image_uuid= self.dataset.image_uuid(self.image_id)
		self.image_metadata= self.dataset.image_metadata(self.image_id)
		t2 = time.time()
		dt_loadimg= t2-t1
		
		#print("image_processed")
		#print(image_processed.shape)
		#print("image_meta")
		#print(image_meta.shape)
		#print(image_meta)

		# - Get detector result
		t1 = time.time()
		inference_output= self.model([np.expand_dims(image_processed, 0), np.expand_dims(image_meta, 0)]) 
		detections, mrcnn_probs, mrcnn_bbox, mrcnn_mask, rpn_rois, rpn_class, rpn_bbox = inference_output
    
		boxes, class_ids, scores, full_masks= utils.reformat_detections(
    	detections=detections[0].numpy(), 
			mrcnn_mask=mrcnn_mask[0].numpy(), 
			original_image_shape=original_input_shape, 
			image_shape=image_processed.shape, 
			window=window
		)
		
		self.class_names= self.dataset.class_names
		self.masks= full_masks
		self.boxes= boxes
		self.class_ids= list(class_ids)
		self.scores= list(scores)
		self.nobjects= self.masks.shape[-1]
		
		#plt.imshow(image_processed)
		#plt.show()
		
		
		masks_proc= mrcnn_mask[0].numpy()
		
		#print("masks_proc")
		#print(masks_proc.shape)
		
		#print("boxes")
		#print(type(boxes))
		#print(boxes.shape)
		#print(boxes)
		#print("class_ids")
		#print(type(class_ids))
		#print(class_ids.shape)
		#print("scores")
		#print(type(scores))
		#print(scores.shape)
		#print("full_masks")
		#print(type(full_masks))
		#print(full_masks.shape)
		
		for i in range(self.nobjects):
			mask= self.masks[:,:,i]
			#print("mask %d, n=%d" % (i+1, np.count_nonzero(mask)))
			#plt.imshow(mask)
			#plt.show()
		
		
		### EXIT DEBUG
		#sys.exit(0)
		##############
		
		#r = self.model.detect([self.image], verbose=0)[0]	
		#self.class_names= self.dataset.class_names
		#self.masks= r['masks']
		#self.boxes= r['rois']
		#self.class_ids= r['class_ids']
		#self.scores= r['scores']
		#self.nobjects= self.masks.shape[-1]
		t2 = time.time()
		dt_modeldet= t2-t1

		# - Remap detected object ids & name?
		t1 = time.time()

		if self.remap_classids and self.classid_map:	
			logger.info("Remapping detection object ids & class names...")		
			class_ids_remapped= []
			
			for class_id in self.class_ids:
				has_remap_classid= class_id in self.classid_map
				
				if has_remap_classid:
					class_id_remap= self.classid_map[class_id]		
					class_ids_remapped.append(class_id_remap)
					logger.info("Remapped id=%d --> id=%d ..." % (class_id,class_id_remap))
				else:
					logger.error("Requested to remap class_id=%d but not found in map keys!" % class_id)
					return -1
					
			self.class_ids= class_ids_remapped

		t2 = time.time()
		dt_remapclass= t2-t1

		# - Retrieve ground truth masks
		t1 = time.time()
		self.class_names_gt= self.dataset.class_names
		self.masks_gt, class_ids_gt_arr= self.dataset.load_mask(self.image_id)
		#self.class_ids_gt= list(class_ids_gt_arr)
		#self.masks_gt= self.dataset.load_gt_masks(self.image_id, binary=False)
		self.class_ids_gt = self.dataset.image_info[self.image_id]["class_ids"]
		self.sidelobes_mixed_or_near_gt = self.dataset.image_info[self.image_id]['sidelobes_mixed_or_near']
		logger.debug("class_ids_gt elements: {}".format(' '.join(map(str, self.class_ids_gt))))

		#print("self.masks_gt.shape")
		#print(self.masks_gt.shape)
		#print("self.class_ids_gt")
		#print(self.class_ids_gt)
		for i in range(self.masks_gt.shape[-1]):
			logger.info("Draw GT masks ...")
			#print("mask_gt.shape")
			#print(self.masks_gt[:,:,i].shape)
			#plt.imshow(self.masks_gt[:,:,i])
			#plt.show()

		self.labels_gt= []
		self.colors_gt= []
		self.captions_gt= []

		for item in self.class_ids_gt:
			label= self.class_names_gt[item]
			color= self.class_color_map[label]
			logger.debug("label_gt=%s" % label)	
			self.labels_gt.append(label)
			self.colors_gt.append(color)
			self.captions_gt.append(label)

		t2 = time.time()
		dt_loadgtmask= t2-t1

		# - Retrieve ground truth object info (available only in input json data)
		t1 = time.time()
		self.objs_gt= self.dataset.load_gt_obj_info(self.image_id)
		if not self.objs_gt:
			logger.warn("gt object info list is empty (hint: no object in this image or input data not in json format)...")
		t2 = time.time()
		dt_loadgtobjinfo= t2-t1

		# - Print elapsed time stats
		tend= time.time()
		dt= tend-tstart

		logger.info("==> get_data() TIME STATS: dt=%.2fs, load_img=%.2f, modeldet=%.2f, remapclass=%.2f, load_gtmask=%.2f, load_gtobjinfo=%.2f" % (dt, dt_loadimg/dt*100., dt_modeldet/dt*100., dt_remapclass/dt*100., dt_loadgtmask/dt*100., dt_loadgtobjinfo/dt*100.))

		return 0

	# ========================
	# ==     PREDICT
	# ========================
	def predict(self, image, image_id='', bboxes_gt=[], header=None, xmin=0, ymin=0):
		""" Predict results on given image """

		# - Throw error if image is None
		if image is None:
			logger.error("No input image given!")
			return -1
			
		self.image= image
		self.image_xmin= xmin
		self.image_ymin= ymin

		if image_id:
			self.image_id= image_id
		if header:
			self.image_header= header
			
		# - Pre-process image?
		image_processed= np.copy(image)
		dp= self.config['preprocess_fcn']
		if dp is not None:
			image_processed= dp(image)
			
		# - Resize image
		image_processed, window, scale, padding, crop = utils.resize_image(
			image_processed,
			min_dim=self.config['image_min_dim'],
			min_scale=self.config['image_min_scale'],
			max_dim=self.config['image_max_dim'],
			mode=self.config['image_resize_mode']
		)	
		
		# - Get image meta
		image_meta= utils.compose_image_meta(
    	image_id=self.image_id,
			original_image_shape=self.image.shape,
			window=window,
			scale=scale,
			active_class_ids=np.zeros([self.config['num_classes']], dtype=np.int32),
			config=self.config
		)
			
		# - Get detector result
		#r = self.model.detect([self.image], verbose=0)[0]
		#self.class_names= self.config.CLASS_NAMES
		#self.masks= r['masks']
		#self.boxes= r['rois']
		#self.class_ids= r['class_ids']
		#self.scores= r['scores']
		#self.nobjects= self.masks.shape[-1]

		inference_output= self.model([np.expand_dims(image_processed, 0), np.expand_dims(image_meta, 0)]) 
		detections, mrcnn_probs, mrcnn_bbox, mrcnn_mask, rpn_rois, rpn_class, rpn_bbox = inference_output
    
		boxes, class_ids, scores, full_masks= utils.reformat_detections(
    	detections=detections[0].numpy(), 
			mrcnn_mask=mrcnn_mask[0].numpy(), 
			original_image_shape=self.image.shape, 
			image_shape=image_processed.shape, 
			window=window
		)
		
		self.class_names= [key for key in self.config['class_dict']] 
		self.masks= full_masks
		self.boxes= boxes
		self.class_ids= list(class_ids)
		self.scores= list(scores)
		self.nobjects= self.masks.shape[-1]

		# - Process detected masks
		if self.nobjects>0:
			logger.info("Processing detected masks for image %s ..." % (self.image_id))
			self.extract_det_masks()
		else:
			logger.warn("No detected object found for image %s ..." % (self.image_id))
			return 0
 
		# - Set gt box if given
		self.bboxes_gt= bboxes_gt
			
		# - Draw results
		if self.draw:
			logger.info("Drawing results for image %s ..." % (str(self.image_id)))
			if self.outfile=="":
				outfile= 'out_' + str(self.image_id) + '.png'
			else:
				outfile= self.outfile
			self.draw_results(outfile)

		# - Create dictionary with detected objects
		self.make_json_results()

		# - Write json results?
		if self.write_to_json:
			logger.info("Writing results for image %s to json ..." % str(self.image_id))
			if self.outfile_json=="":
				outfile_json= 'out_' + str(self.image_id) + '.json'
			else:
				outfile_json= self.outfile_json
			self.write_json_results(outfile_json)

		# - Create DS9 region objects
		self.make_ds9_regions(self.use_polygon_regions)

		# - Write DS9 regions to file?
		if self.write_to_ds9:
			logger.info("Writing detected objects for image %s to DS9 format ..." % str(self.image_id))
			if self.outfile_ds9=="":
				outfile_ds9= 'out_' + str(self.image_id) + '.reg'
			else:
				outfile_ds9= self.outfile_ds9
			self.write_ds9_regions(outfile_ds9)
	
		return 0

	# ========================
	# ==     INSPECT
	# ========================
	def inspect_results(self, image_id, image_path):
		""" Inspect results on given image """
	
		tstart = time.time()
		
		# - Retrieve data from dataset & model
		logger.info("Retrieve data from dataset & model ...")
		t1 = time.time()
		self.image_id= image_id
		self.image_path= image_path
		if self.get_data()<0:
			logger.error("Failed to set data from provided dataset!")
			return -1
		t2 = time.time()
		dt_getdata= t2 - t1 
		logger.debug('==> get_data(): dt=%.2fs' % (t2 - t1))

		# - Process ground truth masks
		logger.info("Processing ground truth masks ...")
		t1 = time.time()
		self.extract_gt_masks()
		t2 = time.time()
		dt_extractgtmasks= t2 - t1 
		logger.debug('==> extract_gt_masks(): dt=%.2fs' % (t2 - t1))


		# - Process detected masks
		dt_extractdetmasks= 0. 
		if self.nobjects>0:
			logger.info("Processing detected masks ...")
			t1 = time.time()
			self.extract_det_masks()
			t2 = time.time()
			dt_extractdetmasks= t2-t1
			logger.debug('==> extract_det_masks(): dt=%.2fs' % (t2 - t1))
		else:
			logger.warn("No detected object found for image %s ..." % self.image_path_base)


		# - Compute morph parameters for detected objects
		dt_computedetmaskpars= 0
		if self.nobjects>0:
			logger.info("Computing morph parameters for detected objects ...")
			t1 = time.time()
			self.compute_det_mask_pars()
			t2 = time.time()
			dt_computedetmaskpars= t2-t1
			logger.debug('==> compute_det_mask_pars(): dt=%.2fs' % (t2 - t1))

		# - Compute performance results
		logger.info("Compute performance results for image %s ..." % self.image_path_base)
		t1 = time.time()
		self.compute_performances()
		t2 = time.time()
		dt_computeperf= t2-t1
		logger.debug('==> compute_performances(): dt=%.2fs' % (t2 - t1))


		# - Draw results
		dt_draw= 0
		if self.draw:
			logger.info("Drawing results for image %s ..." % self.image_path_base)
			t1 = time.time()
			##outfile =  'out_' + self.image_path_base_noext + '.png'
			outfile =  'out_' + self.image_path_base_noext + '_id' + self.image_uuid + '.png'
			self.draw_results(outfile)
			t2 = time.time()
			dt_draw= t2-t1
			logger.debug('==> draw_results(): dt=%.2fs' % (t2 - t1))

		# - Dump time stats
		tend = time.time()
		dt= (tend - tstart)
		logger.info("==> TIME STATS: dt=%.2fs, get_data=%.2f, extract_gt_masks=%.2f, extract_det_masks=%.2f, computedetmaskpars=%.2f, compute_performances=%.2f, draw=%.2f" % (dt, dt_getdata/dt*100., dt_extractgtmasks/dt*100., dt_extractdetmasks/dt*100., dt_computedetmaskpars/dt*100., dt_computeperf/dt*100., dt_draw/dt*100.))
		

		return 0
		
	# ========================
	# ==   EXTRACT GT MASKS
	# ========================
	def extract_gt_masks(self):
		""" Extract ground truth masks & bbox """

		# - Reset gt data
		self.masks_gt_merged= []
		self.class_ids_gt_merged= []
		self.bboxes_gt= []
		self.captions_gt= []
		self.sidelobes_mixed_or_near_gt_merged= []

		# - Split ground truth masks and merge connected ones (if enabled)
		if self.split_gtmasks:

			# - Split non-connected masks for all objects but galaxies
			logger.info("Processing %d ground truth masks to split non-connected ..." % (self.masks_gt.shape[-1]))

			masks_gt_det= []
			class_ids_gt_det= []
			sidelobes_mixed_or_near_gt_det = []
		
			for k in range(self.masks_gt.shape[-1]):
				mask_gt= self.masks_gt[:,:,k]
				label_gt= self.labels_gt[k]
				class_id_gt= self.class_ids_gt[k]
				sidelobe_mixed_or_near_gt= self.sidelobes_mixed_or_near_gt[k]

				if label_gt=='galaxy_C2' or label_gt=='galaxy_C3' or label_gt=='galaxy' or label_gt=='extended' or label_gt=='extended-multisland':
					masks_gt_det.append(mask_gt)
					class_ids_gt_det.append(class_id_gt)
					sidelobes_mixed_or_near_gt_det.append(sidelobe_mixed_or_near_gt)
					continue

				component_labels_gt, ncomponents_gt= self.extract_mask_connected_components(mask_gt)
				logger.debug("Found %d sub components in gt mask no. %d ..." % (ncomponents_gt,k))
		
				for i in range(ncomponents_gt):	
					mask_indices= np.where(component_labels_gt==1)
					extracted_mask= np.zeros(mask_gt.shape,dtype=mask_gt.dtype)
					extracted_mask[mask_indices]= 1

					# - Extract true object id from gt mask pixel values (1=sidelobes,2=sources,3=...)
					#   Override class_id_gt
					# object_classid= mask_gt[mask_indices[0][0],mask_indices[1][0]] # Disabled for the moment
					object_classid= class_id_gt
				
					logger.info("gt mask no. %d (subcomponent no. %d): classid_gt=%d ..." % (k,i,object_classid))

					masks_gt_det.append(extracted_mask)
					class_ids_gt_det.append(object_classid)
					sidelobes_mixed_or_near_gt_det.append(sidelobe_mixed_or_near_gt)
			
			N= len(masks_gt_det)
			g= Graph(N)
			for i in range(N):
				for j in range(i+1,N):
					connected= self.are_mask_connected(masks_gt_det[i],masks_gt_det[j])
					same_class= (class_ids_gt_det[i]==class_ids_gt_det[j])
					mergeable= (connected and same_class)
					if mergeable:
						logger.debug("GT mask (%d,%d) have connected components and can be merged..." % (i,j))
						g.addEdge(i,j)

			cc = g.connectedComponents()
			print("--> Connected components")
			print(cc) 

		
			for i in range(len(cc)):
				if not cc[i]:
					continue
		
				n_merged= len(cc[i])

				for j in range(n_merged):
					index= cc[i][j]
					mask= masks_gt_det[index]
					class_id= class_ids_gt_det[index]
					sidelobe_mixed_or_near_gt= sidelobes_mixed_or_near_gt_det[index]
			
					logger.debug("Merging GT mask no. %d (class_id=%d) ..." % (index,class_id))
					if j==0:
						merged_mask= mask
						#merged_score= score
					else:
						merged_mask= self.merge_masks(merged_mask,mask)
	
				self.masks_gt_merged.append(merged_mask)
				self.class_ids_gt_merged.append(class_id)
				self.sidelobes_mixed_or_near_gt_merged.append(sidelobe_mixed_or_near_gt)
		
		else:
			# - No actions done on input gt masks
			logger.info("#%d true objects present in this image ..." % self.masks_gt.shape[-1])
			for k in range(self.masks_gt.shape[-1]):
				mask_gt= self.masks_gt[:,:,k]
				label_gt= self.labels_gt[k]
				class_id_gt= self.class_ids_gt[k]
				sidelobe_mixed_or_near_gt= self.sidelobes_mixed_or_near_gt[k]
				logger.info("GT mask no. %d: classId=%d, label=%s" % (k,class_id_gt,label_gt))
				self.masks_gt_merged.append(mask_gt)
				self.class_ids_gt_merged.append(class_id_gt)
				self.sidelobes_mixed_or_near_gt_merged.append(sidelobe_mixed_or_near_gt)

		
		# - Compute GT bbox and captions
		for i in range(len(self.masks_gt_merged)):
			mask= self.masks_gt_merged[i]
			height= mask.shape[0]
			width= mask.shape[1]
			mask_expanded = np.zeros([height,width,1],dtype=np.bool)
			mask_expanded[:,:,0]= mask
			bbox= utils.extract_bboxes(mask_expanded)
			self.bboxes_gt.append(bbox[0])
			
			print("BBOX_GT %d" % (i+1))
			print(bbox[0])
	
			label= self.class_names_gt[self.class_ids_gt_merged[i]]
			caption = label
			self.captions_gt.append(caption)	
		

		
	# ========================
	# ==   EXTRACT DET MASKS
	# ========================
	def extract_det_masks(self):
		""" Extract detected masks & bbox """
		
		# - Reset mask data		
		self.masks_final= []
		self.class_ids_final= []
		self.class_names_final= []
		self.scores_final= []	
		self.bboxes= []
		self.captions= []

		# - Select detected objects with score larger than threshold
		N = self.boxes.shape[0]
		masks_sel= []
		class_ids_sel= []
		scores_sel= []
		nobjects_sel= 0
		logger.info("%d objects (%d boxes) found in this image ..." % (self.nobjects,N))

		for i in range(N):
			mask= self.masks[:, :, i]
			class_id = self.class_ids[i]
			score = self.scores[i]
			logger.debug("class_id=%d" % (class_id))
			#print("self.class_names")
			#print(self.class_names)
			
			#print("mask")
			#print(mask)
			#print("class_id")
			#print(class_id)
			#print("score")
			#print(score)
			#print("box")
			#print(self.boxes[i])
			
			#plt.imshow(mask)
			#plt.show()

			label = self.class_names[class_id]
			caption = "{} {:.3f}".format(label, score)
			if score<self.score_thr:
				logger.debug("Skipping object %s (id=%d) with score %f<thr=%f ..." % (label, class_id, score, self.score_thr))
				continue

			logger.debug("Selecting object %s (id=%d) with score %f>thr=%f ..." % (label, class_id, score, self.score_thr))
			masks_sel.append(mask)
			class_ids_sel.append(class_id)
			scores_sel.append(score)
			nobjects_sel+= 1
		
		logger.info("#%d objects selected in this image ..." % len(masks_sel))

		# - Sort objects by descending scores
		sort_indices= np.argsort(scores_sel)[::-1]

		# - Separate all detected objects which are not connected.
		#   NB: This is done only for sources & sidelobes not for galaxies
		masks_det= []
		class_ids_det= []
		scores_det= []
		
		if self.split_masks:

			logger.info("Splitting non-connected detected objects ...")
		
			for index in sort_indices:
				mask= masks_sel[index]	
				class_id= class_ids_sel[index]
				label= self.class_names[class_id]
				score= scores_sel[index]

				# - Skip if class id is galaxy
				if label=='galaxy_C2' or label=='galaxy_C3' or label=='galaxy' or label=='extended-multisland':
					masks_det.append(mask)
					class_ids_det.append(class_id)
					scores_det.append(score)
					logger.debug("Selecting object %s (id=%d) with score %f>thr=%f ..." % (label,class_id,score,self.score_thr))
					continue

				# - Extract components masks
				component_labels, ncomponents= self.extract_mask_connected_components(mask)
				logger.info("Found %d sub components in mask no. %d ..." % (ncomponents,index))
		
				# - Extract indices of components and create masks for extracted components
				for i in range(ncomponents):	
					extracted_mask= np.zeros(mask.shape,dtype=mask.dtype)
					extracted_mask= np.where(component_labels==i+1, [1], [0])

					masks_det.append(extracted_mask)
					class_ids_det.append(class_id)
					scores_det.append(score)
					logger.info("Selecting object %s (id=%d) with score %f>thr=%f ..." % (label,class_id,score,self.score_thr))
			
			logger.info("Found %d components overall (after non-connected component extraction) in this image ..." % (len(masks_det)))
		
		else:
			# Do not split det masks
			for index in sort_indices:
				mask= masks_sel[index]	
				class_id= class_ids_sel[index]
				label= self.class_names[class_id]
				score= scores_sel[index]
				masks_det.append(mask)
				class_ids_det.append(class_id)
				scores_det.append(score)
		
		
		# - Merge connected masks
		masks_merged= []
		class_ids_merged= []
		scores_merged= []

		if self.merge_overlapped_masks:
			# - Init undirected graph
			#   Add links between masks that are connected
			N= len(masks_det)
			g= Graph(N)
			for i in range(N):
				for j in range(i+1,N):
					connected= self.are_mask_connected(masks_det[i],masks_det[j])
					same_class= (class_ids_det[i]==class_ids_det[j])
					mask_iou= jaccard_score(masks_det[i].flatten(), masks_det[j].flatten(), average='binary')
					above_merge_overlap= (mask_iou>=self.merge_overlap_iou_thr)
												
					mergeable= (connected and same_class and above_merge_overlap)
					#mergeable= (connected and same_class)
					if mergeable:
						logger.debug("Detected masks (%d,%d) have connected components and can be merged..." % (i,j))
						g.addEdge(i,j)

			# - Compute connected masks
			cc = g.connectedComponents()
			#print(cc)

			# - Merge connected masks
			for i in range(len(cc)):
				if not cc[i]:
					continue
		
				score_avg= 0
				n_merged= len(cc[i])

				for j in range(n_merged):
					index= cc[i][j]
					mask= masks_det[index]
					class_id= class_ids_det[index]
					score= scores_det[index]
					score_avg+= score

					logger.debug("Merging mask no. %d ..." % index)
					if j==0:
						merged_mask= mask
						merged_score= score
					else:
						merged_mask= self.merge_masks(merged_mask,mask)
	
				score_avg*= 1./n_merged	
				masks_merged.append(merged_mask)
				class_ids_merged.append(class_id)
				scores_merged.append(score_avg)
		
			logger.info("#%d detected object masks left after merging connected ones ..." % len(masks_merged))

		else:
			# - Do not merge connected masks
			for index in range(len(masks_det)):
				mask= masks_det[index]
				class_id= class_ids_det[index]
				score= scores_det[index]
				masks_merged.append(mask)
				class_ids_merged.append(class_id)
				scores_merged.append(score)


		# - Find if there are overlapping masks with different class id
		#   If so retain the one with largest score (using max graph clique)
		if self.select_best_overlapped_masks:

			logger.debug("Inspecting overlapped objects with different ids and retaining the best one (largest score) ...")

			N_final= len(masks_merged)
			#g_final= Graph(N_final)
			g_final= networkx.Graph()

			for i in range(N_final):
				class_id_i= class_ids_merged[i]
				label_i= self.class_names[class_id_i]
				mask_i= masks_merged[i]
				
				for j in range(i+1,N_final):
					class_id_j= class_ids_merged[j]
					label_j= self.class_names[class_id_j]
					mask_j= masks_merged[j]
					connected= self.are_mask_connected(masks_merged[i],masks_merged[j])
					#same_class= (class_id_i==class_id_j)
					###is_sidelobe_other= (label_i=='sidelobe' and label_j!='sidelobe') or (label_i!='sidelobe' and label_j=='sidelobe')
					is_sidelobe_other= (label_i=='spurious' and label_j!='spurious') or (label_i!='spurious' and label_j=='spurious')

					mergeable= connected

					if connected and self.split_source_sidelobe and is_sidelobe_other:
						mask_iou= jaccard_score(mask_i.flatten(), mask_j.flatten(), average='binary')
						if mask_iou<self.merge_overlap_iou_thr:
							logger.info("IOU=%f<%f between overlapping sidelobe-other class, so won't merge in this case..." % (mask_iou,self.merge_overlap_iou_thr))
							mergeable= False						

					if mergeable:
						logger.debug("Merged mask (%d,%d) have connected components and are selected for final selection..." % (i,j))
						#g_final.addEdge(i,j)
						g_final.add_edge(i,j)

			#cc_final= g_final.connectedComponents()
			cc_final= list(networkx.find_cliques(g_final))

			# - Sort cliques by largest component object score
			clique_max_scores= []
			clique_max_score_index= []
			for item in cc_final:
				max_score= -1
				max_score_index= -1
				for index in item: 
					score= scores_merged[index]
					if score>max_score:
						max_score= score
						max_score_index= index
				clique_max_scores.append(max_score)
				clique_max_score_index.append(max_score_index)

			sorted_clique_indices= sorted(range(len(clique_max_scores)), key=lambda k: clique_max_scores[k], reverse=True)

			# - Select objects		
			is_selected= [True]*len(masks_merged)

			for clique_index in sorted_clique_indices:
				index_best= clique_max_score_index[clique_index]
				score_best= clique_max_scores[clique_index]
				class_id_best= class_ids_merged[index_best] 
				
				if is_selected[index_best]:
					logger.debug("Mask with index %d (score=%f, class=%d) selected as the best among all the overlapping masks ..." % (index_best,score_best,class_id_best))

				for index in cc_final[clique_index]: 
					score= scores_merged[index]
					class_id= class_ids_merged[index]
					if index!=index_best and is_selected[index]:
						is_selected[index]= False
						logger.debug("Mask with index %d (score=%f, class=%d) will be excluded ..." % (index,score,class_id))

			# - Compute bounding box, check integrity and set final masks
			for index in range(len(masks_merged)):
				if not is_selected[index]:
					continue
				height= masks_merged[index].shape[0]
				width= masks_merged[index].shape[1]
				mask_expanded = np.zeros([height,width,1],dtype=np.bool)
				mask_expanded[:,:,0]= masks_merged[index]
				bbox= utils.extract_bboxes(mask_expanded)
				
				#print("masks_merged[index]")
				#print(masks_merged[index])
				#print("mask_expanded")
				#print(mask_expanded)
				#print("bbox")
				#print(bbox)
				
				#plt.imshow(masks_merged[index])
				#plt.show()

				if bbox[0][1]>=bbox[0][3] or bbox[0][0]>=bbox[0][2]:
					logger.warn("Invalid det bbox(%d,%d,%d,%d), skip it ..." % (bbox[0][1],bbox[0][3],bbox[0][0],bbox[0][2]) )
					continue

				# - Add to collection
				label= self.class_names[class_ids_merged[index]]
				caption = "{} {:.2f}".format(label, scores_merged[index])

				self.masks_final.append(masks_merged[index])
				self.class_ids_final.append(class_ids_merged[index])
				self.class_names_final.append(label)
				self.scores_final.append(scores_merged[index])
				self.bboxes.append(bbox[0])
				self.captions.append(caption)
		
			logger.info("#%d detected object masks finally selected after selecting best among overlapped ones ..." % len(self.masks_final))

		else:	
			# - Do not select best overlapping objects
			for index in range(len(masks_merged)):
				# - Compute bounding box, check integrity
				height= masks_merged[index].shape[0]
				width= masks_merged[index].shape[1]
				mask_expanded = np.zeros([height,width,1],dtype=np.bool)
				mask_expanded[:,:,0]= masks_merged[index]
				bbox= utils.extract_bboxes(mask_expanded)

				if bbox[0][1]>=bbox[0][3] or bbox[0][0]>=bbox[0][2]:
					logger.warn("Invalid det bbox(%d,%d,%d,%d), skip it ..." % (bbox[0][1],bbox[0][3],bbox[0][0],bbox[0][2]) )
					continue

				# - Add to collection
				label= self.class_names[class_ids_merged[index]]
				caption = "{} {:.2f}".format(label, scores_merged[index])

				self.masks_final.append(masks_merged[index])
				self.class_ids_final.append(class_ids_merged[index])
				self.class_names_final.append(label)
				self.scores_final.append(scores_merged[index])
				self.bboxes.append(bbox[0])
				self.captions.append(caption)

			logger.info("#%d detected object masks finally selected ..." % len(self.masks_final))


	# ============================
	# ==   COMPUTE DET MASK PARS
	# ============================
	def compute_det_mask_pars(self):
		""" Compute detected object mask parameters """

		# - Check image info are available
		has_metadata= True
		if not self.image_metadata:
			has_metadata= False

		if has_metadata:
			nx= self.image_metadata["nx"]
			ny= self.image_metadata["ny"]
			dx= self.image_metadata["dx"]
			dy= self.image_metadata["dy"]
			img_bkg= self.image_metadata["bkg"]
			img_rms= self.image_metadata["rms"]
			bmaj= self.image_metadata["bmaj"]
			bmin= self.image_metadata["bmin"]
			beamArea= np.pi*bmaj*bmin/(4*np.log(2)) # in arcsec^2
			pixelArea= np.abs(dx*dy) # in arcsec^2
			npixInBeam= beamArea/pixelArea
			beamWidth= np.sqrt(np.abs(bmaj*bmin)) # arcsec
			pixScale= np.sqrt(np.abs(dx*dy)) # arcsec
			beamWidthInPixel= int(math.ceil(beamWidth/pixScale))

		# - Read 2d image data (without pre-processing)
		#logger.info("Reading 2D origin image data from file %s ..." % (self.image_path))
		#data, header= utils.read_fits(
		#	self.image_path, 
		#	stretch=False, 
		#	normalize=False, 
		#	convertToRGB=False, 
		#	to_uint8= False,
		#	stretch_biascontrast=False	
		#)
		#nchan= data.shape[-1]
		#ndim= len(data.shape)
		
		nchan= self.image.shape[-1]
		data= self.image
		ndim= len(data.shape)
		if ndim==3:
			data= self.image[:,:,0]
		
		ndim= len(data.shape)
		
		#print("self.image.shape")
		#print(self.image.shape)
		#print("data.shape")
		#print(data.shape)
			
		if ndim!=2:
			logger.error("Image size needed for computing morph pars should be =2 and not %d!" % (ndim))
			return -1

		# - Loop over detected mask and compute pars
		self.det_obj_pars= []

		for i in range(len(self.masks_final)):
			# - Set name
			name= 'Sdet' + str(i+1)

			# - Set class name
			class_name= self.class_names[self.class_ids_final[i]]	

			# - Find contours
			mask= self.masks_final[i]
			bmap= np.copy(mask)
			bmap[bmap>0]= 1
			bmap= bmap.astype(np.uint8)

			logger.debug("Find det obj no. %d contours ..." % (i+1))
			contours= cv.findContours(bmap, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
			contours= imutils.grab_contours(contours)
			logger.debug("#%d contours found ..." % (len(contours)))

			# - Find number of islands
			label_img= measure.label(bmap)

			logger.debug("Find region properties for obj no. %d ..." % (i+1))
			regprops= regionprops(label_image=label_img, intensity_image=data)
			logger.debug("#%d regprops found ..." % (len(regprops)))
			nislands= len(regprops)
			
			# - Find number of pixels 
			cond= np.logical_and(np.isfinite(mask), mask!=0)
			npix_tot= np.count_nonzero(cond)
			
			# - Find signal-to-noise
			data_1d= data[cond]
			Stot= np.nansum(data_1d)
			Sbkg= img_bkg*npix_tot
			S= Stot-Sbkg
			Serr_noise= img_rms*np.sqrt(npix_tot) # NOT SURE THIS IS CORRECT, CHECK!!!
			SNR= S/Serr_noise
			logger.debug("Object no. %d: Stot=%f, S=%f, S_noise=%f, npix=%d, rms=%f, SNR=%f" % (i+1, Stot, S, Serr_noise, npix_tot, img_rms, SNR))
		

			# - Find bounding box of entire object (merging contours)
			#   NB: boundingRect returns Rect(x_top-left, y_top-left, width, height)
			#   NB2: patches.Rectangle wants top-left corner (bottom visually)
			#      top-left means bottom visually as y origin is at the top and is increasing from top to bottom 
			if not contours:
				logger.warn("No contours found for object no. %d, fill empty dict!" % (i+1))
				self.det_obj_pars.append({})
				continue

			for j in range(len(contours)):
				if j==0:
					contours_merged= contours[j]
				else:
					contours_merged= np.append(contours_merged, contours[j], axis=0)
		
			bbox= cv.boundingRect(contours_merged)
			bbox_x_tl= bbox[0] 
			bbox_y_tl= bbox[1]
			bbox_w= bbox[2] 
			bbox_h= bbox[3]
			bbox_x= bbox_x_tl + 0.5*bbox_w
			bbox_y= bbox_y_tl + 0.5*bbox_h
			
			# - Find rotated bounding box of entire object
			bbox_min= cv.minAreaRect(contours_merged)
			bbox_min_x= bbox_min[0][0]
			bbox_min_y= bbox_min[0][1]
			bbox_min_w= bbox_min[1][0] 
			bbox_min_h= bbox_min[1][1]
			bbox_min_angle= bbox_min[2]
			bbox_min_x_tl= bbox_min_x - 0.5*bbox_min_w
			bbox_min_y_tl= bbox_min_y - 0.5*bbox_min_h
		
			bbox_min_points = cv.boxPoints(bbox_min)
		
			
			# - Is at border?
			bbox_norot_x= bbox_x
			bbox_norot_y= bbox_y
			bbox_norot_w= bbox_w
			bbox_norot_h= bbox_h
			bbox_norot_xmin= bbox_norot_x - 0.5*bbox_norot_w
			bbox_norot_xmax= bbox_norot_x + 0.5*bbox_norot_w	
			bbox_norot_ymin= bbox_norot_y - 0.5*bbox_norot_h
			bbox_norot_ymax= bbox_norot_y + 0.5*bbox_norot_h
			at_border_x= (bbox_norot_xmin<=0) or (bbox_norot_xmax>=nx)
			at_border_y= (bbox_norot_ymin<=0) or (bbox_norot_ymax>=ny)
			at_border= (at_border_x or at_border_y)
		
			# - Compute other morph pars
			if has_metadata:
				nbeams= float(npix_tot)/float(npixInBeam)
				minSizeVSBeam= float(min(bbox_min_w,bbox_min_h))/beamWidthInPixel
				maxSizeVSBeam= float(max(bbox_min_w,bbox_min_h))/beamWidthInPixel
				minSizeVSImg= min(float(bbox_norot_w)/float(nx), float(bbox_norot_h)/float(ny))
				maxSizeVSImg= max(float(bbox_norot_w)/float(nx), float(bbox_norot_h)/float(ny))
			else:
				nbeams= -999
				minSizeVSBeam= -999
				maxSizeVSBeam= -999
				minSizeVSImg= -999
				maxSizeVSImg= -999

			# - Fill par dict
			det_obj_par_dict= {}
			det_obj_par_dict["Stot"]= float(Stot)
			det_obj_par_dict["bbox_angle"]= float(bbox_min_angle)
			det_obj_par_dict["bbox_h"]= float(bbox_min_h)
			det_obj_par_dict["bbox_w"]= float(bbox_min_w)
			det_obj_par_dict["bbox_x"]= float(bbox_min_x)
			det_obj_par_dict["bbox_y"]= float(bbox_min_y)
			det_obj_par_dict["border"]= int(at_border)
			det_obj_par_dict["class"]= class_name
			det_obj_par_dict["maxsize_beam"]= maxSizeVSBeam 
			det_obj_par_dict["maxsize_img_fract"]= maxSizeVSImg
			det_obj_par_dict["minsize_beam"]= minSizeVSBeam
			det_obj_par_dict["minsize_img_fract"]= minSizeVSImg
			det_obj_par_dict["name"]= name
			det_obj_par_dict["nbeams"]= nbeams
			det_obj_par_dict["nislands"]= nislands
			det_obj_par_dict["npix"]= npix_tot
			det_obj_par_dict["snr"]= float(SNR)

			print("== det obj pars ==")
			print(det_obj_par_dict)

			self.det_obj_pars.append(det_obj_par_dict)

		return 0


	# ============================
	# ==   COMPUTE PERFORMANCES
	# ============================
	def compute_performances(self):
		""" Compute performances """

		# - Reset matrix
		self.confusion_matrix= np.zeros((self.n_classes,self.n_classes))
		self.confusion_matrix_norm= np.zeros((self.n_classes,self.n_classes))	
		self.purity= np.zeros((1,self.n_classes))
		self.nobjs_true= np.zeros((1,self.n_classes))
		self.nobjs_det= np.zeros((1,self.n_classes))
		self.nobjs_det_right= np.zeros((1,self.n_classes))
		self.detobj_classids= []
		self.detobj_class_names= []
		self.detobj_gtinfo= []
		self.is_gt_obj_detected= []
		self.is_det_obj_matching_to_gt_obj= []
		self.is_det_obj_matching_to_gt_obj_sameclass= []
		self.matchobj_classids= []
		self.matchobj_class_names= []
		self.matchobj_ious= []
		
		# - Loop over gt boxes and find associations to det boxes
		for i in range(len(self.bboxes_gt)):
			# if it is specified not to consider sources near or mixed with sidelobes
			if not self.dataset.consider_sources_near_mixed_sidelobes:
				# and if the current object (source) is near, or mixed with, a sidelobe,
				# do not consider this object in the evaluation
				if self.sidelobes_mixed_or_near_gt_merged[i] == 1:
					continue

			bbox_gt= self.bboxes_gt[i]
			class_id_gt= self.class_ids_gt_merged[i]
			self.nobjs_true[0][class_id_gt]+= 1

			obj_info_gt= {}
			if self.objs_gt and not self.split_gtmasks:
				obj_info_gt= self.objs_gt[i]

			# - Find associations between true and detected objects according to largest IOU
			index_best= -1
			iou_best= 0
			score_best= 0
			logger.debug("len(self.bboxes)=%d, len(self.class_ids_final)=%d" % (len(self.bboxes),len(self.class_ids_final)))

			for j in range(len(self.bboxes)):
				class_id= self.class_ids_final[j]
				score= self.scores_final[j]
				bbox= self.bboxes[j]

				# - Check bbox
				if bbox[1]>=bbox[3] or bbox[0]>=bbox[2]:
					logger.warn("Invalid det bbox (%d,%d,%d,%d) in image %s when computing IOU among boxes, skip it..." % (bbox[1],bbox[3],bbox[0],bbox[2],self.image_path) )
					continue

				if bbox_gt[1]>=bbox_gt[3] or bbox_gt[0]>=bbox_gt[2]:
					logger.warn("Invalid gt bbox (%d,%d,%d,%d) in image %s when computing IOU among boxes, skip it..." % (bbox_gt[1],bbox_gt[3],bbox_gt[0],bbox_gt[2],self.image_path) )
					continue

				iou= addon_utils.get_iou(bbox, bbox_gt)
				mask_iou= jaccard_score(self.masks_final[j].flatten(), self.masks_gt_merged[i].flatten().astype(np.bool), average='binary')

				logger.info("IOU(det=%d,true=%d)=%f, MaskIOU(det=%d,true=%d)=%f" % (j,i,iou,j,i,mask_iou))
				#if iou>=self.iou_thr and iou>=iou_best:
				if mask_iou>=self.iou_thr and mask_iou>=iou_best:
					index_best= j
					#iou_best= iou
					iou_best= mask_iou
					score_best= score

			# - Update confusion matrix and other stats
			if obj_info_gt:
				self.detobj_gtinfo.append(obj_info_gt)

			if index_best==-1:
				logger.info("True object no. %d (class_id=%d) not associated to any detected object ..." % (i+1,class_id_gt))
				self.detobj_scores.append(-999)
				self.detobj_ious.append(-999)
				self.detobj_classids.append(-999)
				self.detobj_class_names.append("not-detected")
				self.is_gt_obj_detected.append(0)
			else:
				class_id_det= self.class_ids_final[index_best]
				self.confusion_matrix[class_id_gt][class_id_det]+= 1
				self.detobj_scores.append(score_best)
				self.detobj_ious.append(iou_best)
				self.detobj_classids.append(class_id_det)
				self.detobj_class_names.append(self.class_names[class_id_det])
				self.is_gt_obj_detected.append(1)
				logger.info("True object no. %d (class_id=%d) associated to detected object no. %d (class_id=%d) ..." % (i+1,class_id_gt,index_best,class_id_det))


		# - Normalize confusion matrix
		for i in range(self.n_classes):
			norm= self.nobjs_true[0][i]
			if norm<=0:
				continue
			for j in range(self.n_classes):
				C= self.confusion_matrix[i][j]
				C_norm= C/norm
				self.confusion_matrix_norm[i][j]= C_norm

		# - Compute purity
		for j in range(len(self.bboxes)):
			bbox= self.bboxes[j]
			class_id= self.class_ids_final[j]
			self.nobjs_det[0][class_id]+= 1

			# - Find association to true box
			index_best= -1
			iou_best= 0
			for i in range(len(self.bboxes_gt)):
				bbox_gt= self.bboxes_gt[i]
				class_id_gt= self.class_ids_gt_merged[i]

				# - Check bbox
				if bbox[1]>=bbox[3] or bbox[0]>=bbox[2]:
					logger.warn("Invalid det bbox (%d,%d,%d,%d) in image %s when computing IOU among boxes, skip it..." % (bbox[1],bbox[3],bbox[0],bbox[2],self.image_path) )
					continue

				if bbox_gt[1]>=bbox_gt[3] or bbox_gt[0]>=bbox_gt[2]:
					logger.warn("Invalid gt bbox (%d,%d,%d,%d) in image %s when computing IOU among boxes, skip it..." % (bbox_gt[1],bbox_gt[3],bbox_gt[0],bbox_gt[2],self.image_path) )
					continue

				iou= addon_utils.get_iou(bbox, bbox_gt)
				mask_iou= jaccard_score(self.masks_final[j].flatten(), self.masks_gt_merged[i].flatten().astype(np.bool), average='binary')

				#if iou>=self.iou_thr and iou>=iou_best:
				if mask_iou>=self.iou_thr and mask_iou>=iou_best:
					index_best= i
					#iou_best= iou
					iou_best= mask_iou

			# - Check if correctly detected
			if index_best!=-1:
				class_id_gt= self.class_ids_gt_merged[index_best]
				self.is_det_obj_matching_to_gt_obj.append(1)
				self.matchobj_classids.append(class_id_gt)
				self.matchobj_class_names.append(self.class_names[class_id_gt])
				self.matchobj_ious.append(iou_best)

				if class_id==class_id_gt:
					self.nobjs_det_right[0][class_id]+= 1
					self.is_det_obj_matching_to_gt_obj_sameclass.append(1)
					#logger.info("Det object %d associated to true object %d (IOU=%f) ..." % (j,index_best,iou_best))
					#print(self.nobjs_det_right)
				else:
					self.is_det_obj_matching_to_gt_obj_sameclass.append(0)

			else:
				self.is_det_obj_matching_to_gt_obj.append(0)
				self.is_det_obj_matching_to_gt_obj_sameclass.append(0)	
				self.matchobj_classids.append(-999)
				self.matchobj_class_names.append("not-matched")
				self.matchobj_ious.append(-999)

		for j in range(self.n_classes):
			if self.nobjs_det[0][j]<=0:
				continue
			p= self.nobjs_det_right[0][j]/self.nobjs_det[0][j]
			self.purity[0][j]= p


		# - Print confusion matrix
		print("== SAMPLE NOBJ TRUE ==")
		print(self.nobjs_true)

		print("== SAMPLE NOBJ DET ==")
		print(self.nobjs_det)

		print("== SAMPLE NOBJ DET CORRECTLY ==")
		print(self.nobjs_det_right)

		print("== SAMPLE CLASSIFICATION MATRIX (not normalized) ==")
		print(self.confusion_matrix)

		print("== SAMPLE CLASSIFICATION MATRIX (normalized) ==")
		print(self.confusion_matrix_norm)

		print("== SAMPLE PRECISION (or PURITY) ==")
		print(self.purity)


	# ====================================
	# ==   WRITE RESULTS IN JSON FORMAT
	# ====================================
	def make_json_results(self):
		""" Create a dictionary with detected objects """
		
		self.results= {
			"image_id": self.image_id, 
			"objs": []
		}

		xmin= self.image_xmin
		ymin= self.image_ymin
		imgshape= self.image.shape
		nx= imgshape[1]
		ny= imgshape[0]

		# - Loop over detected objects
		if self.masks_final:
			for i in range(len(self.masks_final)):
				# - Get detection info
				sname= 'S' + str(i+1) + "_" + self.obj_name_tag
				class_id= self.class_ids_final[i]
				class_name= self.class_names[class_id]
				y1, x1, y2, x2 = self.bboxes[i]
				score= self.scores_final[i]

				x1= int(x1)
				x2= int(x2)
				y1= int(y1)
				y2= int(y2)
				class_id= int(class_id)

				at_edge= False
				if x1<=0 or x1>=nx-1 or x2<=0 or x2>=nx-1:
					at_edge= True
				if y1<=0 or y1>=ny-1 or y2<=0 or y2>=ny-1:
					at_edge= True

				# - Object pixels
				mask= self.masks_final[i]
				pixels= np.argwhere(mask==1).tolist()

				if xmin!=0 or ymin!=0: # add image origin (if not (0,0))
					for npix in range(len(pixels)):
						pixels[npix][0]+= ymin
						pixels[npix][1]+= xmin

				# - Object vertex
				padded_mask = np.zeros( (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
				padded_mask[1:-1, 1:-1] = mask
				contours = find_contours(padded_mask, 0.5)
				vertexes= []
				for verts in contours:
					# - Subtract the padding and flip (y, x) to (x, y)
					verts = np.fliplr(verts) - 1
					vertexes.append(verts.tolist())

				#vertex_list= vertexes[0]
				vertex_list= vertexes
				if xmin!=0 or ymin!=0: # add image origin (if not (0,0))
					for k in range(len(vertex_list)):
						for nvert in range(len(vertex_list[k])):
							vertex_list[k][nvert][0]+= xmin
							vertex_list[k][nvert][1]+= ymin

				d= {
					"name": sname,
					"x1": xmin + x1,
					"x2": xmin + x2,
					"y1": ymin + y1,
					"y2": ymin + y2,
					"class_id": class_id,
					"class_name": class_name,
					"score": score,
					"pixels": pixels,
					"vertexes": vertex_list,
					"edge": at_edge
				}
				self.results["objs"].append(d)
	
		
	def write_json_results(self, outfile):
		""" Write a json file with detected objects """
		
		# - Check if result dictionary is filled
		if not self.results:
			logger.warn("Result obj dictionary is empty, nothing to be written...")
			return
				
		# - Write to file
		with open(outfile, 'w') as fp:
			json.dump(self.results, fp, indent=2, sort_keys=True)
		
	# ====================================
	# ==   WRITE RESULTS IN DS9 FORMAT
	# ====================================
	def make_ds9_regions(self, use_polygon=True):
		""" Make a list of DS9 regions from json results """

		# - Check if result dictionary was created
		self.obj_regions= []
		if not self.results:
			logger.warn("No result dictionary was filled or no object detected, no region will be produced...")
			return -1
		if 'objs' not in self.results:
			logger.warn("No object list found in result dict...")
			return -1

		# - Loop over dictionary of detected object
		for detobj in self.results['objs']:
			sname= detobj['name']
			x1= detobj['x1']
			x2= detobj['x2']
			y1= detobj['y1']
			y2= detobj['y2']
			dx= x2-x1
			dy= y2-y1
			xc= x1 + 0.5*dx
			yc= y1 + 0.5*dy
			class_name= detobj['class_name']
			vertexes= detobj['vertexes']

			# - Set region tag
			at_edge= detobj['edge']
			class_tag= '{' + class_name + '}'

			tags= []
			tags.append(class_tag)
			if at_edge:
				tags.append('{BORDER}')

			color= self.class_color_map_ds9[class_name]
			
			rmeta= regions.RegionMeta({"text": sname, "tag": tags})
			rvisual= regions.RegionVisual({"color": color})

			# - Loop over contours and create one region per contour
			ncontours= len(vertexes)
			for contour in vertexes:			 
				vertexes_x= []
				vertexes_y= []
				for vertex in contour:
					vertexes_x.append(vertex[0])
					vertexes_y.append(vertex[1])
	
				if use_polygon:
					r= regions.PolygonPixelRegion(vertices=regions.PixCoord(x=vertexes_x, y=vertexes_y), meta=rmeta, visual=rvisual)
				else:
					r= regions.RectanglePixelRegion(xc, yc, dx, dy, meta=rmeta, visual= rvisual)
				
				self.obj_regions.append(r)


	def write_ds9_regions(self, outfile):
		""" Write DS9 region file """
	
		# - Check if region list is empty
		if not self.obj_regions:
			logger.warn("Region list with detected objects is empty, nothing to be written...")
			return

		# - Write to file
		try:
			regions.write(filename=outfile, format='ds9', coordsys='image', overwrite=True) # available for version >=0.5
		except:
			try:	
				logger.debug("Failed to write region list to file, retrying with write_ds9 (<0.5 regions API) ...")
				regions.write_ds9(regions=self.obj_regions, filename=outfile, coordsys='image') # this is to be used for versions <0.5 (deprecated in v0.5)
			except Exception as e:
				logger.warn("Failed to write region list to file (err=%s)!" % str(e))
			

	# ========================
	# ==   DRAW RESULTS
	# ========================
	def draw_results(self, outfile):
		""" Draw results """

		# - Create axis
		logger.debug("Create axis...")
		height, width = self.image.shape[:2]
		#figsize=(height,width)
		figsize=(16, 16)
		fig, ax = plt.subplots(1, figsize=figsize)
	
		# - Show area outside image boundaries
		logger.debug("Show area outside image boundaries...")
		title= self.image_path_base_noext
		#ax.set_ylim(height + 10, -10)
		#ax.set_xlim(-10, width + 10)
		ax.set_ylim(height + 2, -2)
		ax.set_xlim(-2, width + 2)
		ax.axis('off')
		#ax.set_title(title,fontsize=30)
	
		#ax.set_frame_on(False)
		
		# - Normalize image for drawing scopes to [0,255]
		masked_image= self.image.astype(np.uint32).copy()
		img_max= masked_image.max()
		if img_max==1:
			masked_image= masked_image*255
		

		# - Draw true bounding box
		if self.bboxes_gt:
			logger.debug("Draw true bounding box...")
			for i in range(len(self.bboxes_gt)):
				label= 'bkg'
				if self.class_ids_gt_merged:
					label= self.class_names[self.class_ids_gt_merged[i]]
				color_gt = self.class_color_map[label]
	
				y1, x1, y2, x2 = self.bboxes_gt[i]
				p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1,alpha=0.7, linestyle="dashed",edgecolor=color_gt, facecolor='none')
				ax.add_patch(p)

				#caption = captions_gt[i]
				caption = ""
				ax.text(x1, y1 + 8, caption, color='w', size=13, backgroundcolor="none")


		# - Draw detected objects
		if self.masks_final:
			logger.debug("Draw detected objects...")
			for i in range(len(self.masks_final)):
				label= self.class_names[self.class_ids_final[i]]
				score= self.scores_final[i]
				color = self.class_color_map[label]
		
				# Bounding box
				y1, x1, y2, x2 = self.bboxes[i]
				dx= x2-x1
				dy= y2-y1
				p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,alpha=0.7, linestyle="solid",edgecolor=color, facecolor='none')
				ax.add_patch(p)
	
				# Label
				if self.draw_class_label_in_caption:
					caption = self.captions[i]
					ax.text(x1, y1 + 8, caption, color=color, size=20, backgroundcolor="none")
				else:
					caption = "{:.2f}".format(score)
					#ax.text(x1 + dx/2 - 4, y1 - 1, caption, color=color, size=23, backgroundcolor="none")
					#ax.text(x1 + dx/2 - 4, y1 - 1, caption, color="mediumturquoise", size=23, backgroundcolor="none")
					ax.text(x1 + dx/2 - 4, y1 - 1, caption, color="darkturquoise", size=30, backgroundcolor="none")

				# Mask
				mask= self.masks_final[i]
				if self.draw_shaded_masks:
					masked_image = visualize.apply_mask(masked_image, mask, color, alpha=0.3)
	
				# Mask Polygon
				# Pad to ensure proper polygons for masks that touch image edges.
				padded_mask = np.zeros( (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
				padded_mask[1:-1, 1:-1] = mask
				contours = find_contours(padded_mask, 0.5)
				for verts in contours:
					# Subtract the padding and flip (y, x) to (x, y)
					verts = np.fliplr(verts) - 1
					p = Polygon(verts, facecolor="none", edgecolor=color)
					ax.add_patch(p)

			# - Draw image
			ax.imshow(masked_image.astype(np.uint8))
	
		else:
			ax.imshow(masked_image)
			##ax.imshow(self.image)

		# - Write to file	
		logger.debug("Write to file %s ..." % outfile)
		t1 = time.time()
		if self.save_plots:
			fig.savefig(outfile)
			##fig.savefig(outfile, bbox_inches='tight')
			plt.close(fig)
			t2 = time.time()
		else:
			t2 = time.time()
			plt.show()

	
	# ========================
	# ==     MASK METHODS
	# ========================
	def merge_masks(self, mask1, mask2):
		""" Merge masks """
		mask= mask1 + mask2
		mask[mask>1]= 1	
		return mask

	def extract_mask_connected_components(self,mask):
		""" Extract mask components """
		labels, ncomponents= skimage.measure.label(mask, background=0, return_num=True, connectivity=1)
		return labels, ncomponents


	def are_mask_connected(self,mask1,mask2):
		""" Check if two masks are connected """

		# - Find how many components are found in both masks
		labels1, ncomponents1= self.extract_mask_connected_components(mask1)
		labels2, ncomponents2= self.extract_mask_connected_components(mask2)
	
		# - Merge masks
		mask= self.merge_masks(mask1,mask2)

		# - Find how many components are found in mask sum
		#   If <ncomp1+ncomp2 the masks are not connected 
		labels, ncomponents= self.extract_mask_connected_components(mask)

		if ncomponents==ncomponents1+ncomponents2:
			connected= False
		else:
			connected= True

		return connected


	
