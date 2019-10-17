# COCO API
Fork of COCO API with extended functionality.

### Installation
pip install git+https://github.com/tillvolkmann/cocoapi.git#subdirectory=PythonAPI

### Data set conversions
#### COCO2VOC
Key info on VOC format:
*Annotations*
* Annotations are store in xml format. An example can be found [here](https://gist.github.com/Prasad9/30900b0ef1375cc7385f4d85135fdb44).
* Each image has its own annotation file. The image file and annotation file should have the same file name, except for the extensions.
* All instances within an image are listed in that single annotation file. Multiple insances of the same object class may be present.

### Dependencies
Following dependencies are only needed for specific data set conversions:
* 2VOC: lxml
* 2tfrecord: tensorflow

### Acknowledgments
This repo builds on the following contributions:
* philferriere/cocoapi
* nightrome/cocostuffapi
* alicranck/coco2voc
* CasiaFan/Dataset_to_VOC_converter
