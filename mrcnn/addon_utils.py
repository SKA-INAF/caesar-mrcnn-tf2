#!/usr/bin/env python

from __future__ import print_function

##################################################
###          MODULE IMPORT
##################################################
## STANDARD MODULES
import sys
import os
import logging
import math
import random
import numpy as np
import functools
import warnings
from distutils.version import LooseVersion
from typing import Tuple

## ASTRO MODULES
from astropy.io import ascii, fits
from astropy.units import Quantity
from astropy.modeling.parameters import Parameter
from astropy.modeling.core import Fittable2DModel
from astropy import wcs
from astropy.wcs import WCS
from astropy import units as u
from astropy.table import Column
from astropy.nddata.utils import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ZScaleInterval, ContrastBiasStretch
from astropy.io.fits.verify import VerifyWarning
warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.simplefilter('ignore', category=VerifyWarning)
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning)

import regions
import fitsio
from fitsio import FITS, FITSHDR

import scipy

##############################
##     GLOBAL VARS
##############################
from mrcnn import logger


## ==================================
## ==    COMPUTE IOU
## ==================================
def get_iou(bb1, bb2):
	""" Calculate the Intersection over Union (IoU) of two bounding boxes """

	bb1_x1= bb1[1]
	bb1_y1= bb1[0]
				 
	bb1_x2= bb1[3]  
	bb1_y2= bb1[2]
 	
	bb2_x1= bb2[1]
	bb2_y1= bb2[0]
		
	bb2_x2= bb2[3]
	bb2_y2= bb2[2]
 	
	assert bb1_x1 < bb1_x2
	assert bb1_y1 < bb1_y2
	assert bb2_x1 < bb2_x2
	assert bb2_y1 < bb2_y2

	# determine the coordinates of the intersection rectangle
	x_left = max(bb1_x1, bb2_x1)
	y_top = max(bb1_y1, bb2_y1)
	x_right = min(bb1_x2, bb2_x2)
	y_bottom = min(bb1_y2, bb2_y2)

	if x_right < x_left or y_bottom < y_top:
		return 0.0

	# The intersection of two axis-aligned bounding boxes is always an
	# axis-aligned bounding box
	intersection_area = (x_right - x_left) * (y_bottom - y_top)

	# compute the area of both AABBs
	bb1_area = (bb1_x2 - bb1_x1) * (bb1_y2 - bb1_y1)
	bb2_area = (bb2_x2 - bb2_x1) * (bb2_y2 - bb2_y1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
	assert iou >= 0.0
	assert iou <= 1.0

	return iou
    
############################################################
#  Data I/O
############################################################
def read_table(filename):
	""" Read ascii table """
	t= ascii.read(filename)
	return t

def get_fits_header(filename):
	""" Read FITS image header """
  
	# - Open file
	try:
		hdu = fits.open(filename, memmap=False)
	except Exception as ex:
		errmsg = 'Cannot read image file: ' + filename
		logger.error(errmsg)
		return None

	# - Get header and check keywords
	header= hdu[0].header
    
	return header



def get_fits_size(filename):
	""" Read FITS image size """

	# - Open file
	try:
		hdu = fits.open(filename, memmap=False)
	except Exception as ex:
		errmsg = 'Cannot read image file: ' + filename
		logger.error(errmsg)
		return None

	# - Get header and check keywords
	header= hdu[0].header
	if 'NAXIS1' not in header:
		logger.error("NAXIS1 keyword missing in header!")
		return None
	if 'NAXIS2' not in header:
		logger.error("NAXIS2 keyword missing in header!")
		return None

	nx= header['NAXIS1']
	ny= header['NAXIS2']
    
	return nx, ny


def read_fits(filename, strip_deg_axis=False):
	""" Read FITS image and return data """

	# - Open file
	try:
		hdu= fits.open(filename, memmap=False)
	except Exception as ex:
		errmsg= 'Cannot read image file: ' + filename
		logger.error(errmsg)
		return None

	# - Read data
	data= hdu[0].data
	data_size= np.shape(data)
	nchan= len(data.shape)
	if nchan==4:
		output_data= data[0,0,:,:]
	elif nchan==2:
		output_data= data	
	else:
		errmsg= 'Invalid/unsupported number of channels found in file ' + filename + ' (nchan=' + str(nchan) + ')!'
		logger.error(errmsg)
		hdu.close()
		return None
		
	# - Replace NANs with zeros
	output_data[~np.isfinite(output_data)]= 0

	# - Read metadata
	header= hdu[0].header

	# - Strip degenerate axis
	if strip_deg_axis:
		header= strip_deg_axis_from_header(header)

	# - Get WCS
	wcs = WCS(header)
	if wcs is None:
		logger.warn("No WCS in input image!")
	
	# - Close file
	hdu.close()

	return output_data, header, wcs


	
def strip_deg_axis_from_header(header):
	""" Remove references to 3rd & 4th axis from FITS header """
	
	# - Remove 3rd axis
	if 'NAXIS3' in header:
		del header['NAXIS3']
	if 'CTYPE3' in header:
		del header['CTYPE3']
	if 'CRVAL3' in header:
		del header['CRVAL3']
	if 'CDELT3' in header:
		del header['CDELT3']
	if 'CRPIX3' in header:
		del header['CRPIX3']
	if 'CUNIT3' in header:
		del header['CUNIT3']
	if 'CROTA3' in header:
		del header['CROTA3']
	if 'PC1_3' in header:
		del header['PC1_3']
	if 'PC01_03' in header:
		del header['PC01_03']
	if 'PC2_3' in header:
		del header['PC2_3']
	if 'PC02_03' in header:
		del header['PC02_03']
	if 'PC3_1' in header:
		del header['PC3_1']
	if 'PC03_01' in header:
		del header['PC03_01']
	if 'PC3_2' in header:
		del header['PC3_2']
	if 'PC03_02' in header:
		del header['PC03_02']
	if 'PC3_3' in header:
		del header['PC3_3']
	if 'PC03_03' in header:
		del header['PC03_03']

	# - Remove 4th axis
	if 'NAXIS4' in header:
		del header['NAXIS4']
	if 'CTYPE4' in header:
		del header['CTYPE4']
	if 'CRVAL4' in header:
		del header['CRVAL4']
	if 'CDELT4' in header:
		del header['CDELT4']
	if 'CRPIX4' in header:
		del header['CRPIX4']
	if 'CUNIT4' in header:
		del header['CUNIT4']
	if 'CROTA4' in header:
		del header['CROTA4']
	if 'PC1_4' in header:
		del header['PC1_4']
	if 'PC01_04' in header:
		del header['PC01_04']
	if 'PC2_4' in header:
		del header['PC2_4']
	if 'PC02_04' in header:
		del header['PC02_04']
	if 'PC3_4' in header:
		del header['PC3_4']
	if 'PC03_04' in header:
		del header['PC03_04']
	if 'PC4_1' in header:
		del header['PC4_1']
	if 'PC04_01' in header:
		del header['PC04_01']
	if 'PC4_2' in header:
		del header['PC4_2']
	if 'PC04_02' in header:
		del header['PC04_02']
	if 'PC4_3' in header:
		del header['PC4_3']
	if 'PC04_03' in header:
		del header['PC04_03']
	if 'PC4_4' in header:
		del header['PC4_4']
	if 'PC04_04' in header:
		del header['PC04_04']

	# - Set naxis to 2
	header['NAXIS']= 2
	
	return header
		
		
		
def read_fits_crop(filename, ixmin, ixmax, iymin, iymax, strip_deg_axis=False):
	""" Read a portion of FITS image specified by x-y ranges and return data. Using fitsio module and not astropy. NB: xmax/ymax pixel are excluded """

	# - Check if entire tile has to be read
	read_full= (ixmin==0 or ixmin==-1) and (ixmax==0 or ixmax==-1) and (iymin==0 or iymin==-1) and (iymax==0 or iymax==-1)
	if read_full:
		logger.warn("Reading entire image as given image ranges are all <=0 (not an error if this is the user intention)...")
		return read_fits(filename, strip_deg_axis)
		
	# - Check tile ranges given
	if ixmin<0 or ixmax<0: 
		logger.error("ixmin/ixmax must be >0")
		return None
		
	if iymin<0 or iymax<0: 
		logger.error("iymin/iymax must be >0")
		return None
		
	if ixmax<=ixmin:
		logger.error("ixmax must be >ixmin!")
		return None
		
	if iymax<=iymin:
		logger.error("iymax must be >iymin!")
		return None
	
	# - Open file
	try:
		f= fitsio.FITS(filename)
	except Exception as e:
		logger.error("Failed to open file %s (err=%s)!" % (filename, str(e)))
		return None
		
	# - Read image chunk
	hdu_id= 0
	data_dims= f[hdu_id].get_dims()
	nchan= len(data_dims)
	try:
		if nchan==4:
			data= f[hdu_id][0, 0, iymin:iymax, ixmin:ixmax]
			data= data[0,0,:,:]
		elif nchan==2:
			data= f[hdu_id][iymin:iymax, ixmin:ixmax]
		else:
			logger.error("Invalid/unsupported number of channels (nchan=%d) found in file %s!" % (nchan, filename))
			f.close()
			return None

	except Exception as e:
		logger.error("Failed to read data in range[%d:%d,%d:%d] from file %s (err=%s)!" % (iymin, iymax, ixmin, ixmax, filename, str(e)))
		f.close()
		return None
		
	# - Replace NANs with zeros
	data[~np.isfinite(data)]= 0
		
	# - Read header
	header = f[hdu_id].read_header()
	
	# - Strip degenerate axis
	if strip_deg_axis:
		header= strip_deg_axis_from_header(header)

	# - Get WCS
	wcs = WCS(header)
	if wcs is None:
		logger.warn("No WCS in input image!")
		
	# - Close file
	f.close()

	return data, header, wcs




def apply_mask(image, mask, color, alpha=0.5):
	"""Apply the given mask to the image """
	for c in range(3):
		image[:, :, c] = np.where(	
			mask == 1,
    	image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
			image[:, :, c]
		)
	return image



def resize_img(image, output_shape, order=1, mode='constant', cval=0, clip=True, preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
	""" A wrapper for Scikit-Image resize().

			Scikit-Image generates warnings on every call to resize() if it doesn't receive the right parameters. The right parameters depend on the version 
			of skimage. This solves the problem by using different parameters per version. And it provides a central place to control resizing defaults.
	"""
	if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
		# New in 0.14: anti_aliasing. Default it to False for backward
		# compatibility with skimage 0.13.
		return skimage.transform.resize(
			image, output_shape,
			order=order, mode=mode, cval=cval, clip=clip,
			preserve_range=preserve_range, anti_aliasing=anti_aliasing,
			anti_aliasing_sigma=anti_aliasing_sigma
		)
	else:
		return skimage.transform.resize(
			image, output_shape,
			order=order, mode=mode, cval=cval, clip=clip,
			preserve_range=preserve_range
		)

	
def resize_img_v2(image, min_dim=None, max_dim=None, min_scale=None, mode="square", order=1, anti_aliasing=False, preserve_range=True):
	""" Resizes an image keeping the aspect ratio unchanged.

			Inputs:
				min_dim: if provided, resizes the image such that it's smaller dimension == min_dim
				max_dim: if provided, ensures that the image longest side doesn't exceed this value.
				min_scale: if provided, ensure that the image is scaled up by at least this percent even if min_dim doesn't require it.    
				mode: Resizing mode:
					none: No resizing. Return the image unchanged.
					square: Resize and pad with zeros to get a square image of size [max_dim, max_dim].
					pad64: Pads width and height with zeros to make them multiples of 64. If min_dim or min_scale are provided, it scales the image up before padding. max_dim is ignored.     
					crop: Picks random crops from the image. First, scales the image based on min_dim and min_scale, then picks a random crop of size min_dim x min_dim. max_dim is not used.
				order: Order of interpolation (default=1=bilinear)
				anti_aliasing: whether to use anti-aliasing (suggested when down-scaling an image)

			Returns:
				image: the resized image
				window: (y1, x1, y2, x2). If max_dim is provided, padding might
					be inserted in the returned image. If so, this window is the
					coordinates of the image part of the full image (excluding
					the padding). The x2, y2 pixels are not included.
				scale: The scale factor used to resize the image
				padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
	"""
    
	#image= img_as_float64(image)
	#image= img_as_float(image)
		
	# Keep track of image dtype and return results in the same dtype
	image_dtype = image.dtype
	image_ndims= image.ndim
		
	# - Default window (y1, x1, y2, x2) and default scale == 1.
	h, w = image.shape[:2]
	window = (0, 0, h, w)
	scale = 1
	if image_ndims==3:
		padding = [(0, 0), (0, 0), (0, 0)] # with multi-channel images
	elif image_ndims==2:
		padding = [(0, 0)] # with 2D images
	else:
		logger.error("Unsupported image ndims (%d), returning None!" % (image_ndims))
		return None
	
	crop = None

	if mode == "none":
		return image, window, scale, padding, crop

	# - Scale?
	if min_dim:
		# Scale up but not down
		scale = max(1, min_dim / min(h, w))

	if min_scale and scale < min_scale:
		scale = min_scale

	# Does it exceed max dim?
	if max_dim and mode == "square":
		image_max = max(h, w)
		if round(image_max * scale) > max_dim:
			scale = max_dim / image_max

	# Resize image using bilinear interpolation
	if scale != 1:
		#print("DEBUG: Resizing image from size (%d,%d) to size (%d,%d) (scale=%d)" % (h,w,round(h * scale),round(w * scale),scale))
		image = resize_img(image, (round(h * scale), round(w * scale)), preserve_range=preserve_range, order=order, anti_aliasing=anti_aliasing)

	# Need padding or cropping?
	if mode == "square":
		# Get new height and width
		h, w = image.shape[:2]
		top_pad = (max_dim - h) // 2
		bottom_pad = max_dim - h - top_pad
		left_pad = (max_dim - w) // 2
		right_pad = max_dim - w - left_pad

		if image_ndims==3:
			padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)] # multi-channel
		elif image_ndims==2:
			padding = [(top_pad, bottom_pad), (left_pad, right_pad)] # 2D images
		else:
			logger.error("Unsupported image ndims (%d), returning None!" % (image_ndims))
			return None

		image = np.pad(image, padding, mode='constant', constant_values=0)
		window = (top_pad, left_pad, h + top_pad, w + left_pad)

	elif mode == "pad64":
		h, w = image.shape[:2]
		# - Both sides must be divisible by 64
		if min_dim % 64 != 0:
			logger.error("Minimum dimension must be a multiple of 64, returning None!")
			return None

		# Height
		if h % 64 > 0:
			max_h = h - (h % 64) + 64
			top_pad = (max_h - h) // 2
			bottom_pad = max_h - h - top_pad
		else:
			top_pad = bottom_pad = 0
		
		# - Width
		if w % 64 > 0:
			max_w = w - (w % 64) + 64
			left_pad = (max_w - w) // 2
			right_pad = max_w - w - left_pad
		else:
			left_pad = right_pad = 0

		if image_ndims==3:
			padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
		elif image_ndims==2:
			padding = [(top_pad, bottom_pad), (left_pad, right_pad)]
		else:
			logger.error("Unsupported image ndims (%d), returning None!" % (image_ndims))
			return None

		image = np.pad(image, padding, mode='constant', constant_values=0)
		window = (top_pad, left_pad, h + top_pad, w + left_pad)
    
	elif mode == "crop":
		# - Pick a random crop
		h, w = image.shape[:2]
		y = random.randint(0, (h - min_dim))
		x = random.randint(0, (w - min_dim))
		crop = (y, x, min_dim, min_dim)
		image = image[y:y + min_dim, x:x + min_dim]
		window = (0, 0, min_dim, min_dim)
    
	else:
		logger.error("Mode %s not supported!" % (mode))
		return None
    
	return image.astype(image_dtype), window, scale, padding, crop


def resize_mask(mask: np.ndarray, scale: float, padding: int, crop: Tuple[int, int, int, int] = None) -> np.ndarray:
	"""
		Resizes a mask using the given scale and padding.
		Typically, you get the scale and padding from resize_image() to 
		ensure both, the image and the mask, are resized consistently.

		scale: mask scaling factor
		padding: Padding to add to the mask in the form [(top, bottom), (left, right), (0, 0)]
	"""
        
	# Suppress warning from scipy 0.13.0, the output shape of zoom() is
	# calculated with round() instead of int()
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
	if crop is not None:
		y, x, h, w = crop
		mask = mask[y:y + h, x:x + w]
	else:
		mask = np.pad(mask, padding, mode='constant', constant_values=0)
        
	return mask

################################
## PARALLEL SPLITTING UTILS
################################
def generate_tiles(img_xmin, img_xmax, img_ymin, img_ymax, tileSizeX, tileSizeY, gridStepSizeX, gridStepSizeY):
  """ Generate coordinates of image tiles """

  # - Check given arguments
  if img_xmax<=img_xmin:
    logger.error("xmax must be > xmin!")
    return None
  if img_ymax<=img_ymin:
    logger.error("ymax must be > ymin!")
    return None
  if tileSizeX<=0 or tileSizeY<=0:
    logger.error("Invalid box size given!")
    return None
  if gridStepSizeX<=0 or gridStepSizeY<=0 or gridStepSizeX>1 or gridStepSizeY>1:
    logger.error("Invalid grid step size given (null or negative)!")
    return None

  # - Check if image size is smaller than required box
  Nx= img_xmax - img_xmin + 1
  Ny= img_ymax - img_ymin + 1
  
  if tileSizeX>Nx or tileSizeY>Ny:
    logger.warn("Invalid box size given (too small or larger than image size)!")
    return None

  stepSizeX= int(np.round(gridStepSizeX*tileSizeX))
  stepSizeY= int(np.round(gridStepSizeY*tileSizeY))

  # - Generate x & y min & max coordinates
  indexX= 0
  indexY= 0
  ix_min= []
  ix_max= []
  iy_min= []
  iy_max= []

  while indexY<=Ny:
    #offsetY= min(tileSizeY-1,Ny-1-indexY)
    offsetY= min(tileSizeY,Ny-indexY)
    ymin= indexY
    ymax= indexY + offsetY
    #ymax= indexY + offsetY + 1

    if ymin>=Ny or offsetY==0:
      break
    iy_min.append(ymin)
    iy_max.append(ymax)
    indexY+= stepSizeY

  #print("iy min/max")
  #print(iy_min)
  #print(iy_max)

  while indexX<=Nx:
    #offsetX= min(tileSizeX-1,Nx-1-indexX)
    offsetX= min(tileSizeX,Nx-indexX)
    xmin= indexX
    xmax= indexX + offsetX
    #xmax= indexX + offsetX + 1
    if xmin>=Nx or offsetX==0: 
      break
    ix_min.append(xmin)
    ix_max.append(xmax)
    indexX+= stepSizeX

  #print("ix min/max")
  #print(ix_min)
  #print(ix_max)

  # - Generate tile coordinates as tuple list
  tileGrid= []
  for j in range(len(iy_min)):
    for i in range(len(ix_min)):
      tileGrid.append( (img_xmin+ix_min[i], img_xmin+ix_max[i], img_ymin+iy_min[j], img_ymin+iy_max[j]) )

  return tileGrid

################################
## CODE UTILS
################################
def compose_fcns(*funcs):
  """ Compose a list of functions like (f . g . h)(x) = f(g(h(x)) """
  return functools.reduce(lambda f, g: lambda x: f(g(x)), funcs)
  
def set_type(s: str) -> str:
	""" 
		Move float and int types from 64 to 32 bits
			Args:
				s: str, data type name
			Returns: str, new data type
	"""
	mapping = {
		'int64': 'int32',
		'int32': 'int32',
		'float32': 'float32',
		'float64': 'float32'
	}
	s = mapping[s] if s in mapping.keys() else s
	return s  

