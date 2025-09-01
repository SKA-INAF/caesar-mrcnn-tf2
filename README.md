# caesar-mrcnn-tf2
Radio source instance segmentation with Mask R-CNN.    
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
