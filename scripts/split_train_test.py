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
import copy

from sklearn.model_selection import train_test_split

## USER MODULES
from mrcnn import logger


############################################################
#        PARSE/VALIDATE ARGS
############################################################

def parse_args():
	""" Parse command line arguments """  
  
	# - Parse command line arguments
	parser = argparse.ArgumentParser(description='Mask R-CNN options')

	parser.add_argument('--inputfile', dest='inputfile', required=True, type=str, help='Input file to be splitted in train/test sets') 
	parser.add_argument('--test_data_fract', dest='test_data_fract', required=False, default=0.4, help='Fraction of input data used for test (default=0.4)')
	
	args = parser.parse_args()

	return args

############################################################
#        CREATE TRAIN/VAL SETS
############################################################
def create_train_val_sets_from_filelist(filelist, crossval_size=0.4, train_filename='train.dat', crossval_filename='crossval.dat'):
	""" Read input filelist with format img,mask,label and create train & val filelists """
	
	# - Read input list
	data= []
	with open(filelist,'r') as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			data.append(line)

	# - Return train/cval filenames
	return create_train_val_sets_from_list(data, crossval_size, train_filename, crossval_filename)


def create_train_val_sets_from_list(data, crossval_size=0.4, train_filename='train.dat', crossval_filename='crossval.dat'):
	""" Read filelist with format img,mask,label and create train & val filelists """
	
	# - Check if list is empty
	nentries= len(data)
	if nentries<=0:
		logger.error("Given filelist is empty!")
		return []
	if nentries<10:
		logger.warn("Given filelist contains less than 10 entries ...")
	
	# - Shuffle and split train/val sets
	random.shuffle(data)
	x_train, x_crossval = train_test_split(data, test_size=float(crossval_size))
	
	# - Write both sets to files
	logger.info("Writing #%d entries to training dataset list ..." % len(x_train))
	with open(train_filename, 'w') as f:
		for item in x_train:
			f.write("%s\n" % item)

	logger.info("Writing #%d entries to cross-validation dataset list ..." % len(x_crossval))
	with open(crossval_filename, 'w') as f:
		for item in x_crossval:
			f.write("%s\n" % item)

	# - Return filenames
	return [train_filename, crossval_filename]
	
	
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
	#==   SPLIT DATA
	#===========================
	logger.info("Creating file data split ...")
	create_train_val_sets_from_filelist(
		args.inputfile,
		args.test_data_fraction
	)
		
		
return 0


###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())	
	
