#!/usr/bin/python

__author__ = 'tillvolkmann'


from pycocotools.coco import COCO
import os
import json
from PIL import Image
import argparse


def resize_coco(annotation_file, output_annotation_file,
                image_dir=None, output_image_dir=None,
                output_image_size=[None, None], preseve_aspect=True,
                output_file_type=None):
    """
    Resizes images in COCO dataset, and annotations accordingly

    :param annotation_file:
    :param output_annotation_file:
    :param image_dir:
    :param output_image_dir:
    :param output_image_size:
    :param preseve_aspect:
    :param output_file_type:
    """

    assert output_file_type.lower() in ['png', 'jpeg']

    # create instance of coco object
    coco_instance = COCO(annotation_file)
    coco_imgs = coco_instance.imgs

    # get lists of anns and imgs for indexing
    list_ann_ids = [anno['id'] for anno in coco_instance.dataset['annotations']]
    list_img_ids = [img_id for img_id in coco_imgs]

    # make dirs
    output_annotation_dir = os.path.dirname(output_annotation_file)
    if not os.path.exists(output_annotation_dir):
        os.makedirs(output_annotation_dir)
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

        # infer original extension from image
    if output_file_type is None:
        output_file_type = os.path.splitext(coco_imgs[list(coco_imgs.keys())[0]]['file_name'])[1][1:].lower()
    # Pillow requires 'jpeg' as format spec, but 'jpg' is more common; so we pass 'jpeg' to pillow and use 'jpg' for file name
    output_file_type_str = output_file_type
    if output_file_type_str.lower() == 'jpeg':
        output_file_type_str = 'jpg'

    # process each image
    for i, img_id in enumerate(coco_imgs):

        # get anns
        anns_ids = coco_instance.getAnnIds(img_id)

        # get image meta
        img_meta = coco_imgs[img_id]
        img_width, img_height = img_meta['width'], img_meta['height']

        # get output width and height
        output_img_width, output_img_height = output_image_size[0], output_image_size[1]
        if preseve_aspect:
            resize_ratio_width = float(output_img_width) / float(img_width)
            resize_ratio_height = float(output_img_height) / float(img_height)
            if resize_ratio_width < resize_ratio_height:  # this implementation makes sure max sizes are matched exactly
                resize_ratio_height = resize_ratio_width
                output_img_height = int(round(img_height * resize_ratio_height, 0))
            else:
                resize_ratio_width = resize_ratio_height
                output_img_width = int(round(img_width * resize_ratio_width, 0))

        # get again actual ratios after all rounding is done
        resize_ratio_width = float(output_img_width) / float(img_width)
        resize_ratio_height = float(output_img_height) / float(img_height)
        # resize image
        if image_dir is not None:
            img_name = img_meta["file_name"]
            img_file_name = os.path.join(image_dir, img_name)
            output_img_file_name = os.path.join(output_image_dir,
                                                os.path.splitext(img_name)[0] + '.' + output_file_type_str)
            img = Image.open(img_file_name)
            img = img.resize(
                (output_img_width, output_img_height))  # Image.ANTIALIAS was preferred in PIL, but not in Pillow
            img.save(output_img_file_name, format=output_file_type, quality=95, optimize=True, progressive=False)  # for export settings, see: https://pillow.readthedocs.io/en/5.1.x/handbook/image-file-formats.html

        # resize annotations
        for ann_id in anns_ids:
            # get index into anns from ann id
            index = list_ann_ids.index(ann_id)
            # resize bboxes
            coco_instance.dataset['annotations'][index]['bbox'][0] *= resize_ratio_width
            coco_instance.dataset['annotations'][index]['bbox'][1] *= resize_ratio_height
            coco_instance.dataset['annotations'][index]['bbox'][2] *= resize_ratio_width
            coco_instance.dataset['annotations'][index]['bbox'][3] *= resize_ratio_height
            # resize segmentation
            for s in range(
                    len(coco_instance.dataset['annotations'][index]['segmentation'])):  # for each segmentation fragment
                # multiply every other value by respective scaling factor
                coco_instance.dataset['annotations'][index]['segmentation'][s][::2] \
                    = [val * resize_ratio_width for val in
                       coco_instance.dataset['annotations'][index]['segmentation'][s][::2]]
                coco_instance.dataset['annotations'][index]['segmentation'][s][1::2] \
                    = [val * resize_ratio_width for val in
                       coco_instance.dataset['annotations'][index]['segmentation'][s][1::2]]
            # resize area
            coco_instance.dataset['annotations'][index]['area'] *= resize_ratio_width * resize_ratio_height

        # adjust stored image width/heigth
        index = list_img_ids.index(img_id)
        coco_instance.dataset['images'][index]['width'] = output_img_width
        coco_instance.dataset['images'][index]['height'] = output_img_height

    # write annotation data set
    with open(output_annotation_file, 'w') as json_file:
        json.dump(coco_instance.dataset, json_file)

    # print termination message
    print("Successfully resized COCO data set!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_file", type=str, help="annotation file for object instance/keypoint")
    parser.add_argument("--output_annotation_file", type=str, help="output_annotation_file")
    parser.add_argument("--image_dir", type=str, default=None, help="object instance or keypoint")
    parser.add_argument("--output_image_dir", type=str,default=None,  help="output directory for voc annotation xml file")
    parser.add_argument("--output_image_size", type=list, default=None, help="output directory for voc annotation xml file")
    parser.add_argument("--preseve_aspect", type=bool, default=True, help="whether to preserve aspect ratio")
    parser.add_argument("--output_file_type", type=str, choices=['png', 'jpeg'], default=None, help="output file type")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    resize_coco(args.annotation_file, args.output_annotation_file,
                args.image_dir, args.output_image_dir,
                args.output_image_size, args.preseve_aspect,
                args.output_file_type)
