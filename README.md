# caesar-mrcnn-tf2
Radioastronomical object detector tool based on Mask R-CNN instance segmentation framework.     
This software is a refactorization of `https://github.com/SKA-INAF/caesar-mrcnn.git` for TensorFlow v2.x. 

## **Credit**
This software is distributed with GPLv3 license. If you use it for your research, please add a reference to this github repository and acknowledge these works in your paper:   

* S. Riggi et al., *Astronomical source detection in radio continuum maps with deep neural networks*, 2023, Astronomy and Computing, 42, 100682, [doi](https://doi.org/10.1016/j.ascom.2022.100682)    

## **Installation**  

To build and install the package:    

* Download the software in a local directory, e.g. ```SRC_DIR```:   
  ```$ git clone https://github.com/SKA-INAF/caesar-mrcnn-tf2.git```   
* Create and activate a virtual environment, e.g. ```caesar-mrcnn-tf2```, under a desired path ```VENV_DIR```     
  ```$ python3 -m venv $VENV_DIR/caesar-mrcnn-tf2```    
  ```$ source $VENV_DIR/caesar-mrcnn-tf2/bin/activate```   
* Install dependencies inside venv:   
  ```(caesar-mrcnn-tf2)$ pip install -r $SRC_DIR/requirements.txt```   
* Build and install package in virtual env:   
  ```(caesar-mrcnn-tf2)$ python setup.py install```    
       
To use package scripts:

* Add binary directory to your ```PATH``` environment variable:   
  ``` export PATH=$PATH:$VENV_DIR/caesar-yolo/bin ```    

## **Usage**  
The software can be run using the provided script ```run.py``` in different modes:   

* To train a model: ```(caesar-mrcnn-tf2)$ python $VENV_DIR/bin/run.py [OPTIONS] train```      
* To test a model: ```(caesar-mrcnn-tf2)$ python $VENV_DIR/bin/run.py [OPTIONS] test```    
* To detect objects on new data: ```(caesar-mrcnn-tf2)$ python $VENV_DIR/bin/run.py [OPTIONS] inference```    

Supported options are:  

**INPUT DATA**  
  `--datalist=[VALUE]`: Path to train/test data filelist containing a list of json files. Default: ''     
  `--datalist_val=[VALUE]`: Path to validation data filelist containing a list of json files. Default: ''     
  `--maxnimgs=[VALUE]`: Max number of images to consider in dataset (-1=all). Default: -1   
  `--skip_classes`: Skip certain classes when loading dataset. Default: disabled   
  `--skipped_classes=[VALUE]`: List of class names to be skipped in data loading'. Default: 'compact'        
  `--require_classes`: Require at least one oject class of `required_classes` to be present in image to consider this input data in data loading. Default: disabled   
  `--required_classes=[VALUE]`: List of required object classes. Default: 'extended,extended-multisland,flagged,spurious'    
  
**MODEL**    
  `--weights=[VALUE]`: Path to model weights .h5 file. Default: ''    
  `--backbone_weights=[VALUE]`: Backbone network initialization weights: {random, imagenet, Path to weights .h5 file}. Default: 'random'     
  `--classdict=[VALUE]`: Class id dictionary used when loading dataset. Default: '{"sidelobe":1,"source":2,"galaxy":3}'     
  `--classdict_model=[VALUE]`: Class id dictionary used for the model (if empty, it is set equal to classdict). Default: ''    
  `--remap_classids`: Size in pixel used to resize input image. Default: 256     
  `--classid_remap_dict=[VALUE]`: Dictionary used to remap detected classid to gt classid. Default: ''      

**DATA PRE-PROCESSING**     
  `--imgsize=[VALUE]`: Size in pixel used to resize input image. Default: 256     
  `--normalize_minmax`: Normalize each channel in range [norm_min, norm_max]. Default: no normalization    
  `--norm_min=[VALUE]`: Normalization min value. Default: 0.0    
  `--norm_max=[VALUE]`: Normalization max value. Default: 1.0   
  `--subtract_bkg`: Subtract bkg from ref channel image. Default: no subtraction  
  `--sigma_bkg=[VALUE]`: Sigma clip value used in bkg calculation. Default: 3.0  
  `--use_box_mask_in_bkg`: Compute bkg value in borders left from box mask. Default: not used   
  `--bkg_box_mask_fract=[VALUE]`: Size of mask box dimensions with respect to image size used in bkg calculation. Default: 0.7   
  `--bkg_chid=[VALUE]`: Channel used to subtract background (-1=all). Default: -1   
  `--clip_shift_data`: Apply sigma clip shifting. Default: not applied   
  `--sigma_clip=[VALUE]`: Sigma threshold to be used for clip & shifting pixels. Default: 1.0    
  `--clip_data`: Apply sigma clipping. Default: not applied    
  `--sigma_clip_low=[VALUE]`: Lower sigma threshold to be used for clipping pixels below (mean - sigma_low x stddev). Default: 10.0   
  `--sigma_clip_up=[VALUE]`: Upper sigma threshold to be used for clipping pixels above (mean + sigma_up x stddev). Default: 10.0     
  `--clip_chid=[VALUE]`: Channel used to clip data (-1=all). Default: -1     
  `--zscale_stretch`: Apply zscale transform to data. Default: not applied    
  `--zscale_contrasts=[VALUES]`: zscale contrasts applied to all channels, separated by commas. Default: 0.25,0.25,0.25        
  `--chan3_preproc`: Use the 3-channel pre-processor. Default: not used          
  `--sigma_clip_baseline=[VALUE]`: Lower sigma threshold to be used for clipping pixels below (mean - sigma_low x stddev) in first channel of 3-channel preprocessing. Default: 0.0         
  `--nchannels=[VALUE]`: Number of channels. If you modify channels in preprocessing you must set this option accordingly. Default: 1        

**DATA AUGMENTATION**   
  `--use_augmentation`: Run data augmentation on input images. Default: disabled  
  `--augmenter=[VALUE]`: Augmenter version to be used {"v1","v2","v3"}. "v1" is equal to TF1 caesar-mrcnn. Default: "v1"  

**MODEL TRAINING**    
  `--nepochs=[VALUE]`: Number of training epochs. Default: 1  
  `--rpn_anchor_scales=[VALUES]`: RPN anchor scales in pixels (5 comma-separated values). Default: '4,8,16,32,64'  
  `--max_gt_instances=[VALUE]`: Max GT instances. Default: 300  
  `--backbone=[VALUE]`: Backbone network {resnet101,resnet50,custom}. Default: resnet101  
  `--freeze_backbone`: Freeze backbone weights. Default: free  
  `--backbone_strides=[VALUES]`: Backbone strides in pixels (5 comma-separated values). Default: '4,8,16,32,64'   
  `--rpn_nms_threshold=[VALUE]`: RPN Non-Maximum-Suppression threshold. Default: 0.7  
  `--rpn_train_anchors_per_image=[VALUE]`: Number of anchors per image to use for RPN training. Default: 512  
  `--train_rois_per_image=[VALUE]`: Number of ROIs per image to feed to classifier/mask heads. Default: 512  
  `--rpn_anchor_ratios=[VALUES]`: RPN anchor ratios, comma separated. Default: '0.5,1,2'  
  `--rpn_class_loss_weight=[VALUE]`: RPN classification loss weight. Default: 1.0  
  `--rpn_bbox_loss_weight=[VALUE]`: RPN bounding box loss weight. Default: 1.0  
  `--mrcnn_class_loss_weight=[VALUE]`: Classification loss weight. Default: 1.0  
  `--mrcnn_bbox_loss_weight=[VALUE]`: Bounding box loss weight. Default: 1.0  
  `--mrcnn_mask_loss_weight=[VALUE]`: Mask loss weight. Default: 1.0  
  `--rpn_class_loss`: Enable RPN classification loss.   
  `--no_rpn_class_loss`: Disable RPN classification loss.     
  `--rpn_bbox_loss`: Enable RPN box loss.   
  `--no_rpn_bbox_loss`: Disable RPN box loss.    
  `--mrcnn_class_loss`: Enable classification loss.    
  `--no_mrcnn_class_loss`: Disable classification loss.     
  `--mrcnn_bbox_loss`: Enable box loss.    
  `--no_mrcnn_bbox_loss`: Disable box loss.     
  `--mrcnn_mask_loss`: Enable mask loss.     
  `--no_mrcnn_mask_loss`: Disable mask loss.    
  `--no_l2reg_loss`: Disable L2 regularization loss.    
  `--weight_classes`: Enable weighting of object classes.    
  `--optimizer=[VALUE]`: Optimizer {sgd,adam,adamax}. Default: sgd     
  `--learning_rate=[VALUE]`: Learning rate. Default: 0.0005     
  `--opt_momentum=[VALUE]`: Momentum parameter in SGD. Default: 0.9    
  `--opt_clipnorm=[VALUE]`: clipnorm optimizer parameter. Default: 5.0    
  `--opt_clipvalue=[VALUE]`: clipvalue optimizer parameter. Default: None             	
 
**RUN**     
   `--ngpu`: Number of GPUs used. Default: 1    
   `--nimg_per_gpu`: Number of images per gpu. Default: 1        
  
	
	
