from coco2voc_seg import *
from coco2voc_ann import *
from PIL import Image

import argparse, json
import os, re


def convert_coco2voc(annotation_file, image_dir, output_dir, type='instance', copy_images=False):
    """
    # !!Change paths to your local machine!!
     = '/home/dl/1TB-Volumn/MSCOCO2017/annotations/instances_train2017.json'
    output_dir = '/home/dl/PycharmProjects/coco2voc-master/output'
    image_dir = '/home/dl/1TB-Volumn/MSCOCO2017/train2017'
    """
    assert type in ['instance', 'keypoint']

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # Convert COCO annotations to VOC segmentation masks png images
    coco2voc_seg(annotation_file, output_dir, n=None, compress=True)
    # Convert COCO annotations to VOC annotation xml
    coco2voc_ann(annotation_file, output_dir, type='instance'):
    # Copy images to output_dir
    if copy_images:
        # specific output dir for a split is named after "images_" plus name of the split
        output_dir_images = os.path.join(output_dir, "JPEGImages")
        # create output directory if the leaf directory does not exist
        if not os.path.exists(output_dir_images):
            os.makedirs(output_dir_images)
        # get source image_paths from master dict
        for
        image_dir
        split_image_ids = [image["id"] for image in coco_split_dic[split]["images"]]
        # move each image
        for image_id in split_image_ids:
            # get path from master dict
            image_path = master_file_dic[image_id]['image_file_path']
            # create target path
            target_path = os.path.join(out_dir_images, os.path.basename(image_path))
            io.copy_file(image_path, target_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_file", help="annotation file for object instance/keypoint")
    parser.add_argument("--image_dir", help="image dir")
    parser.add_argument("--type", type=str, help="object instance or keypoint", choices=['instance', 'keypoint'])
    parser.add_argument("--output_dir", help="output directory for voc annotation xml file")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    convert_coco2voc(args.anno_file, args.image_dir, args.output_dir, args.type)
