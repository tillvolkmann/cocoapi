from pycocotools import mask as maskUtils
import numpy as np
import cytoolz
from lxml import etree, objectify
import os, re

def annsToSeg(anns, coco_instance):
    '''
    converts COCO-format annotations of a given image to a PASCAL-VOC segmentation style label
     !!!No guarantees where segmentations overlap - might lead to loss of objects!!!
    :param anns: COCO annotations as returned by 'coco.loadAnns'
    :param coco_instance: an instance of the COCO class from pycocotools
    :return: three 2D numpy arrays where the value of each pixel is the class id, instance number, and instance id.
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
    '''

    coco_instance = COCO(anns_file)
    coco_imgs = coco_instance.imgs

    if n is None:
        n = len(coco_imgs)
    else:
        assert type(n) == int, "n must be an int"
        n = min(n, len(coco_imgs))

    instance_target_path = os.path.join(target_folder, 'instance_labels')
    class_target_path = os.path.join(target_folder, 'class_labels')
    id_target_path = os.path.join(target_folder, 'id_labels')

    os.makedirs(instance_target_path, exist_ok=True)
    os.makedirs(class_target_path, exist_ok=True)
    os.makedirs(id_target_path, exist_ok=True)

    image_id_list = open(os.path.join(target_folder, 'images_ids.txt'), 'a+')
    start = time.time()

    for i, img in enumerate(coco_imgs):

        anns_ids = coco_instance.getAnnIds(img)
        anns = coco_instance.loadAnns(anns_ids)
        if not anns:
            continue

        class_seg, instance_seg, id_seg = annsToSeg(anns, coco_instance)

        Image.fromarray(class_seg).convert("L").save(class_target_path + '/' + str(img) + '.png')
        Image.fromarray(instance_seg).convert("L").save(instance_target_path + '/' + str(img) + '.png')
        
        if compress:
            np.savez_compressed(os.path.join(id_target_path, str(img)), id_seg)
        else:
            np.save(os.path.join(id_target_path, str(img)+'.npy'), id_seg)

        image_id_list.write(str(img)+'\n')

        if i%100==0 and i>0:
            print(str(i)+" annotations processed" +
                  " in "+str(int(time.time()-start)) + " seconds")
        if i>=n:
            break

    image_id_list.close()
    return


def instance2xml_base(anno):
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('VOC2014_instance/{}'.format(anno['category_id'])),
        E.filename(anno['file_name']),
        E.source(
            E.database('MS COCO 2014'),
            E.annotation('MS COCO 2014'),
            E.image('Flickr'),
            E.url(anno['coco_url'])
        ),
        E.size(
            E.width(anno['width']),
            E.height(anno['height']),
            E.depth(3)
        ),
        E.segmented(0),
    )
    return anno_tree


def instance2xml_bbox(anno, bbox_type='xyxy'):
    """bbox_type: xyxy (xmin, ymin, xmax, ymax); xywh (xmin, ymin, width, height)"""
    assert bbox_type in ['xyxy', 'xywh']
    if bbox_type == 'xyxy':
        xmin, ymin, w, h = anno['bbox']
        xmax = xmin+w
        ymax = ymin+h
    else:
        xmin, ymin, xmax, ymax = anno['bbox']
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.object(
        E.name(anno['category_id']),
        E.bndbox(
            E.xmin(xmin),
            E.ymin(ymin),
            E.xmax(xmax),
            E.ymax(ymax)
        ),
        E.difficult(anno['iscrowd'])
    )
    return anno_tree


def parse_instance(content, outdir):
    categories = {d['id']: d['name'] for d in content['categories']}
    # merge images and annotations: id in images vs image_id in annotations
    merged_info_list = list(map(cytoolz.merge, cytoolz.join('id', content['images'], 'image_id', content['annotations'])))
    # convert category id to name
    for instance in merged_info_list:
        instance['category_id'] = categories[instance['category_id']]
    # group by filename to pool all bbox in same file
    for name, groups in cytoolz.groupby('file_name', merged_info_list).items():
        anno_tree = instance2xml_base(groups[0])
        # if one file have multiple different objects, save it in each category sub-directory
        filenames = []
        for group in groups:
            filenames.append(os.path.join(outdir, re.sub(" ", "_", group['category_id']),
                                    os.path.splitext(name)[0] + ".xml"))
            anno_tree.append(instance2xml_bbox(group, bbox_type='xyxy'))
        for filename in filenames:
            etree.ElementTree(anno_tree).write(filename, pretty_print=True)
        print("Formating instance xml file {} done!".format(name))


def keypoints2xml_base(anno):
    annotation = etree.Element("annotation")
    etree.SubElement(annotation, "folder").text = "VOC2014_keypoints"
    etree.SubElement(annotation, "filename").text = anno['file_name']
    source = etree.SubElement(annotation, "source")
    etree.SubElement(source, "database").text = "MS COCO 2014"
    etree.SubElement(source, "annotation").text = "MS COCO 2014"
    etree.SubElement(source, "image").text = "Flickr"
    etree.SubElement(source, "url").text = anno['coco_url']
    size = etree.SubElement(annotation, "size")
    etree.SubElement(size, "width").text = str(anno["width"])
    etree.SubElement(size, "height").text = str(anno["height"])
    etree.SubElement(size, "depth").text = '3'
    etree.SubElement(annotation, "segmented").text = '0'
    return annotation


def keypoints2xml_object(anno, xmltree, keypoints_dict, bbox_type='xyxy'):
    assert bbox_type in ['xyxy', 'xywh']
    if bbox_type == 'xyxy':
        xmin, ymin, w, h = anno['bbox']
        xmax = xmin+w
        ymax = ymin+h
    else:
        xmin, ymin, xmax, ymax = anno['bbox']
    key_object = etree.SubElement(xmltree, "object")
    etree.SubElement(key_object, "name").text = anno['category_id']
    bndbox = etree.SubElement(key_object, "bndbox")
    etree.SubElement(bndbox, "xmin").text = str(xmin)
    etree.SubElement(bndbox, "ymin").text = str(ymin)
    etree.SubElement(bndbox, "xmax").text = str(xmax)
    etree.SubElement(bndbox, "ymax").text = str(ymax)
    etree.SubElement(key_object, "difficult").text = '0'
    keypoints = etree.SubElement(key_object, "keypoints")
    for i in range(0, len(keypoints_dict)):
        keypoint = etree.SubElement(keypoints, keypoints_dict[i+1])
        etree.SubElement(keypoint, "x").text = str(anno['keypoints'][i*3])
        etree.SubElement(keypoint, "y").text = str(anno['keypoints'][i*3+1])
        etree.SubElement(keypoint, "v").text = str(anno['keypoints'][i*3+2])
    return xmltree


def parse_keypoints(content, outdir):
    keypoints = dict(zip(range(1, len(content['categories'][0]['keypoints'])+1), content['categories'][0]['keypoints']))
    # merge images and annotations: id in images vs image_id in annotations
    merged_info_list = map(cytoolz.merge, cytoolz.join('id', content['images'], 'image_id', content['annotations']))
    # convert category name to person
    for keypoint in merged_info_list:
        keypoint['category_id'] = "person"
    # group by filename to pool all bbox and keypoint in same file
    for name, groups in cytoolz.groupby('file_name', merged_info_list).items():
        filename = os.path.join(outdir, os.path.splitext(name)[0]+".xml")
        anno_tree = keypoints2xml_base(groups[0])
        for group in groups:
            anno_tree = keypoints2xml_object(group, anno_tree, keypoints, bbox_type="xyxy")
        doc = etree.ElementTree(anno_tree)
        doc.write(open(filename, "w"), pretty_print=True)
        print("Formating keypoints xml file {} done!".format(name))


def coco2voc_ann(annotation_file, output_dir, type='instance'):
    """
    Convert COCO annotations to VOC annotation xml
    """
    assert type in ['instance', 'keypoint']

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if type == 'instance':
        content = json.load(open(annotation_file, 'r'))
        # make subdirectories
        sub_dirs = [re.sub(" ", "_", cate['name']) for cate in content['categories']]
        for sub_dir in sub_dirs:
            sub_dir = os.path.join(output_dir, str(sub_dir))
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
        parse_instance(content, output_dir)
    
    elif type == 'keypoint':
        parse_keypoints(content, output_dir)