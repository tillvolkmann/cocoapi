from pycocotools.coco2voc_seg import *
from pycocotools.coco2voc_ann import *
from PIL import Image
import argparse, json
import os
from pycocotools.ioutils import list_files_in_dir, copy_file


def coco2voc(annotation_file, image_dir, output_dir, ann_type='instance', copy_images=False):
    """
        Example call:
    annotation_file = '/home/user/data/coco/annotations/instances_test.json'
    image_dir = '/home/user/data/coco/images_test'
    output_dir = '/home/user/data/voc_test'

    VOC2012 contains:
    - Annotations: xml format
    - ImageSets:
    - JPEGImages: a folder with no subfolders, containing all images of the data set in jpeg format
    - SegmentationClass: a folder with no subfolders, containing png files ..
    - SegmentationClassRaw: same as SegmentationClass, but ...
    - SegmentationObject:

    :param annotation_file:
    :param image_dir:
    :param output_dir:
    :param ann_type:
    :param copy_images: copies image to sub dir "JPEGImages" in output_dir
    :return:
    """
    assert ann_type in ['instance', 'keypoint']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert COCO annotations to VOC segmentation masks png images
    coco2voc_seg(annotation_file, output_dir, n=None, compress=True)

    # Convert COCO annotations to VOC annotation xml
    coco2voc_ann(annotation_file, output_dir, type=ann_type)

    # copy images into output directory
    if copy_images:
        # make output dir
        output_dir_images = os.path.join(output_dir, "JPEGImages")
        if not os.path.exists(output_dir_images):
            os.makedirs(output_dir_images)
        # get source image_paths
        source_image_paths = list_files_in_dir(image_dir, extension=['png', 'jpg', 'jpeg'], sub_dirs=False)
        # copy each image
        for source_image_path in source_image_paths:
            destination_image_path = os.path.join(output_dir_images, os.path.basename(source_image_path))
            copy_file(source_image_path, destination_image_path)

    # print termination message
    print("Successfully converted COCO data set to VOC format!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_file", help="annotation file for object instance/keypoint")
    parser.add_argument("--image_dir", help="image dir")
    parser.add_argument("--anno_type", type=str, help="object instance or keypoint", choices=['instance', 'keypoint'])
    parser.add_argument("--output_dir", help="output directory for voc annotation xml file")
    parser.add_argument("--copy_images", help="whether to copy images to output_dir")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    coco2voc(args.anno_file, args.image_dir, args.output_dir, args.anno_type, args.copy_images)
