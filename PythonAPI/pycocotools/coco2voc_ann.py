import argparse, json
import cytoolz
from lxml import etree, objectify
import os, re

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


def parse_instance_by_category(content, outdir):
    """
    Original version, does create replicate files for each category.
    """
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


def parse_instance(content, outdir):
    """
    Version that does not create replicate files for each category.
    """
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
        filename = os.path.join(outdir, os.path.splitext(name)[0] + ".xml")
        for group in groups:
            anno_tree.append(instance2xml_bbox(group, bbox_type='xyxy'))
        etree.ElementTree(anno_tree).write(filename, pretty_print=True)


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


def coco2voc_ann(annotation_file, output_dir, type='instance', separate_categories=False):
    """
    Convert COCO annotations to VOC annotation xml

    :param separate_categories: if True, replicate xmls into separate folders for each category, where each folder
        contains only those xmls that contain the respective category, albeit with the remaining categories still in that xml.
        If False, will create only one xml per image in folder "Annotations".

    Credit to: https://github.com/CasiaFan/Dataset_to_VOC_converter/blob/master/anno_coco2voc.py
    The default behavior of above implentation is to save xml files containing all classes but into
    separate folders by class. This is the non-default behavior in this implementation,
    but can be achieved by setting separate_categories=True.
    """
    assert type in ['instance', 'keypoint']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if type == 'instance':
        print("Formatting instance xml files ...")
        content = json.load(open(annotation_file, 'r'))
        if separate_categories:
            # make subdirectories, one for each category
            sub_dirs = [re.sub(" ", "_", cate['name']) for cate in content['categories']]
            for sub_dir in sub_dirs:
                sub_dir = os.path.join(output_dir, str(sub_dir))
                if not os.path.exists(sub_dir):
                    os.makedirs(sub_dir)
            parse_instance_by_category(content, output_dir)
        else:
            # make a single annotations folder, called "Annotations" according with original VOC2012
            sub_dir = os.path.join(output_dir, 'Annotations')
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            parse_instance(content, sub_dir)

    elif type == 'keypoint':  # note there is no diff here between categories, so separating is not relevant
        print("Formatting keypoint xml files ...")
        parse_keypoints(content, output_dir)

