VOC2012 contains:
* Annotations: xml format
* ImageSets:
* JPEGImages: a folder with no subfolders, containing all images of the data set in jpeg format
* SegmentationClass: a folder with no subfolders, containing png files ..
* SegmentationClassRaw: same as SegmentationClass, but ...
* SegmentationObject:

Details as follows:

**Annotations**


**ImageSets**

These are a couple folders containing text files listing image identifiers for task-specific subsets of VOC.
For instance, the VOC2012/ImageSets/Segmentation/ directory contains text files specifying lists of images for the segmentation task. The files train.txt, val.txt, trainval.txt and test.txt list the image identifiers for the corresponding image sets (training, validation, training+validation and testing). Each line of the file contains a single image identifier. For example, the first three lines of train.txt:
```
2007_000032
2007_000039
2007_000063
...
```

**JPEGImages**

Contains all original RGB images.

**SegmentationClass**

These are class segmentation mask images. Each pixel represents an index.
index 0 corresponds to background and index 255 corresponds to 'void' or unlabelled.
All other indices correspond to classes in alphabetical order (1=aeroplane, 2=bicycle, and so on). Example segmentations can be seen [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html).

These original images are palettised. See useful discussion e.g. [here](https://stackoverflow.com/questions/51676447/python-use-pil-to-load-png-file-gives-strange-results?noredirect=1#comment90316222_51676447). So, rather than storing a full RGB triplet for each pixel, there is a list of 256 RGB triplets (i.e., a colormap) and then at each pixel location it just stores the index into the list. This saves space.

Essentially, these are 8-bit images cosisting in background (0) and an integer valued segmentation index (1-21, as there are 21 classes). A colormap can be accessed through this index for display. Additionally, a hull of not-attributed pixels exist around each object (value 255). These are pixels that are not attributed to either an object or background, and ignored in the evaluation.


**SegmentationClassRaw**

Same as SegmentationClass masks, but with the colormap removed.

See also [download_and_convert_voc2012.sh](https://github.com/tensorflow/models/blob/master/research/deeplab/datasets/download_and_convert_voc2012.sh) in Deeplab repo, and the function [remove_gt_colormap.py](https://github.com/tensorflow/models/blob/1af55e018eebce03fb61bba9959a04672536107d/research/deeplab/datasets/remove_gt_colormap.py).

Applying aforementioned remove_gt_colormap.py to remove the colormap yields these masks. Essentially, just convert the seg images to a numpy array, then save as image.

If you look at the file_download_and_convert_voc2012.sh, there are lines marked by "# Remove the colormap in the ground truth annotations". This part process the original SegmentationClass files and produce the raw segmented image files, which have each pixel value between 0 : 20. (If you may ask why, check this post: Python: Use PIL to load png file gives strange results)
