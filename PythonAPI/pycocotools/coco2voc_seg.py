from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
import numpy as np
import cytoolz
from lxml import etree, objectify
import os, re
from PIL import Image
import matplotlib.pyplot as plt
import time
from pycocotools.voclabelcolormap import color_map

def annsToSeg(anns, coco_instance):
    '''
    converts COCO-format annotations of a given image to a PASCAL-VOC segmentation style label
     !!!No guarantees where segmentations overlap - might lead to loss of objects!!!
    :param anns: COCO annotations as returned by 'coco.loadAnns'
    :param coco_instance: an instance of the COCO class from pycocotools
    :return: three 2D numpy arrays where the value of each pixel is the class id, instance number, and instance id,
        respectively.
    '''
    image_details = coco_instance.loadImgs(anns[0]['image_id'])[0]

    h = image_details['height']
    w = image_details['width']

    class_seg = np.zeros((h, w))
    instance_seg = np.zeros((h, w))
    id_seg = np.zeros((h, w))
    masks, anns = annsToMask(anns, h, w)

    for i, mask in enumerate(masks):
        class_seg = np.where(class_seg>0, class_seg, mask*anns[i]['category_id'])
        instance_seg = np.where(instance_seg>0, instance_seg, mask*(i+1))
        id_seg = np.where(id_seg > 0, id_seg, mask * anns[i]['id'])

    return class_seg, instance_seg, id_seg.astype(np.int64)


def annToRLE(ann, h, w):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = ann['segmentation']
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann['segmentation']
    return rle


def annsToMask(anns, h, w):
    """
    Convert annotations which can be polygons, uncompressed RLE, or RLE to binary masks.
    :return: a list of binary masks (each a numpy 2D array) of all the annotations in anns
    """
    masks = []
    anns = sorted(anns, key=lambda x: x['area'])  # Smaller items first, so they are not covered by overlapping segs
    for ann in anns:
        rle = annToRLE(ann, h, w)
        m = maskUtils.decode(rle)
        masks.append(m)
    return masks, anns


def coco2voc_seg(anns_file, target_folder, type="instance", n=None, compress=True):
    '''
    This function converts COCO style annotations to PASCAL VOC style instance and class
        segmentations. Additionaly, it creates a segmentation mask(1d ndarray) with every pixel contatining the id of
        the instance that the pixel belongs to.
    :param anns_file: COCO annotations file, as given in the COCO data set
    :param Target_folder: path to the folder where the results will be saved
    :param n: Number of image annotations to convert. Default is None in which case all of the annotations are converted
    :param compress: if True, id segmentation masks are saved as '.npz' compressed files. if False they are saved as '.npy'
    :return: All segmentations are saved to the target folder, along with a list of ids of the images that were converted

    Credit to:
    '''

    assert type == "instance", NotImplementedError("Only type 'instance' is implemented")

    coco_instance = COCO(anns_file)
    coco_imgs = coco_instance.imgs

    if n is None:
        n = len(coco_imgs)
    else:
        assert isinstance(n, int), "n must be an int"
        n = min(n, len(coco_imgs))

    # set and create output dirs
    instance_target_path = os.path.join(target_folder, 'SegmentationInstance')  # 'instance_labels')
    classcolor_target_path = os.path.join(target_folder, 'SegmentationClass')  # 'class_labels')
    class_target_path = os.path.join(target_folder, 'SegmentationClassRaw')  # 'class_labels')
    id_target_path = os.path.join(target_folder, 'SegmentationId')  # 'id_labels')
    list_target_path = os.path.join(target_folder, 'ImageSets/Segmentation')

    os.makedirs(instance_target_path, exist_ok=True)
    os.makedirs(classcolor_target_path, exist_ok=True)
    os.makedirs(class_target_path, exist_ok=True)
    os.makedirs(id_target_path, exist_ok=True)
    os.makedirs(list_target_path, exist_ok=True)

    # get VOC palette (color map)
    cmap = color_map()

    # instantiate image id and name list
    image_id_list = open(os.path.join(list_target_path,
                                        f'images_ids_{os.path.splitext(os.path.basename(anns_file))[0]}.txt'), 'a+')   # not sure if this is needed
    image_name_list = open(os.path.join(list_target_path,
                                        f'image_names_{os.path.splitext(os.path.basename(anns_file))[0]}.txt'), 'a+')
    start = time.time()

    print("Creating VOC segmentation masks ...")

    for i, img_id in enumerate(coco_imgs):

        # get anns
        anns_ids = coco_instance.getAnnIds(img_id)
        anns = coco_instance.loadAnns(anns_ids)

        # skip if no anns
        if not anns:
            continue

        # get class, instance, and id segmentation arrays
        class_seg, instance_seg, id_seg = annsToSeg(anns, coco_instance)

        # get image name
        img_name = coco_imgs[img_id]['file_name']

        # convert class segmentation images
        Image.fromarray(class_seg).convert("L").save(os.path.join(class_target_path, img_name))
        # convert instance segmentation images
        Image.fromarray(instance_seg).convert("L").save(os.path.join(instance_target_path, img_name))
        # convert id segmentation images
        if compress:
            np.savez_compressed(os.path.join(id_target_path, img_name), id_seg)
        else:
            np.save(os.path.join(id_target_path, img_name + '.npy'), id_seg)

        # make a seg map equivalent to original VOC segs
        tmp_img = Image.fromarray(class_seg).convert("L")
        tmp_img.putpalette(cmap)
        tmp_img.save(os.path.join(classcolor_target_path, img_name))

        # append to image id list
        image_id_list.write(str(img_id)+'\n')
        image_name_list.write(os.path.splitext(os.path.basename(img_name))[0]+'\n')

        # print status
        if not (i+1) % 100:
            print(f"processed {str(i)} of {n} annotations in {str(int(time.time()-start))} seconds")

        # exit if n exceeded
        if i >= n:
            break

    image_id_list.close()
    return


