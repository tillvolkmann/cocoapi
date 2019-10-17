#!/usr/bin/python

__author__ = 'tillvolkmann'

# Resizes images in COCO dataset, and annotations accordingly

import os
from pycocotools import mask
from pycocotools.cocostuffhelper import cocoSegmentationToPng
from pycocotools.coco import COCO
import skimage.io
import matplotlib.pyplot as plt

def resizecoco(data_root='', out_dir='export_png', img_out_size=None, scale_factor=None):
    '''
    Converts COCO segmentation .json files (GT or results) to one .png file per image.
    :param data_root: location of the COCO root folder
    :param ann_file: identifier of the ground-truth annotation file
    :param res_file: identifier of the result annotation file (if any)
    :param out_dir: the name of the subfolder where we store .png images
    :param is_annotation: whether the COCO file is a GT annotation or a result file
    :param exportImageLimit: limit number of images to be processed; set to "0" for no limit
    :return: None
    '''

    # Define paths
    annPath = f'{data_root}/annotations/{ann_file}'
    if is_annotation:
        pngFolder = '%s/annotations/%s' % (data_root, out_dir)
    else:
        pngFolder = '%s/results/%s' % (data_root, out_dir)
        resPath = '%s/results/stuff_%s_results.json' % (data_root, res_file)

    # Create output folder
    if not os.path.exists(pngFolder):
        os.makedirs(pngFolder)

    # Initialize COCO ground-truth API
    coco = COCO(annPath)
    imgIds = coco.getImgIds()

    # Initialize COCO result
    if not is_annotation:
        coco = coco.loadRes(resPath)
        imgIds = sorted(set([a['image_id'] for a in coco.anns.values()]))

    # Limit number of images
    if exportImageLimit < len(imgIds) and exportImageLimit != 0:
        imgIds = imgIds[0:exportImageLimit]

    # Convert each image to a png
    imgCount = len(imgIds)
    for i in xrange(0, imgCount):
        imgId = imgIds[i]
        imgName = coco.loadImgs(ids=imgId)[0]['file_name'].replace('.jpg', '.png')  # note this will have no effect if images are already in png
        print('Exporting image %d of %d: %s' % (i+1, imgCount, imgName))
        segmentationPath = '%s/%s' % (pngFolder, imgName)
        cocoSegmentationToPng(coco, imgId, segmentationPath)

    # Visualize the last image
    originalImage = skimage.io.imread(coco.loadImgs(imgId)[0]['coco_url'])
    segmentationImage = skimage.io.imread(segmentationPath)
    plt.figure()
    plt.subplot(121)
    plt.imshow(originalImage)
    plt.axis('off')
    plt.title('original image')

    plt.subplot(122)
    plt.imshow(segmentationImage)
    plt.axis('off')
    plt.title('annotated image')
    plt.show()

if __name__ == "__main__":
    cocoseg2png()
