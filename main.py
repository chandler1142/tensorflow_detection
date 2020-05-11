#%%
import os, glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras


tf.random.set_seed(2234)
np.random.seed(2234)



#%%
print(tf.__version__)
# print(tf.config.list_physical_devices('GPU'))
print(tf.config.list_logical_devices())

# %%
import xml.etree.ElementTree as ET

def parse_annotation(img_dir, ann_dir, labels):

    imgs_info = []
    max_boxes = 0
    for ann in os.listdir(ann_dir):
        tree = ET.parse(os.path.join(ann_dir, ann))

        img_info = dict()
        img_info['object'] = []
        boxes_counter = 0
        for elem in tree.iter():
            if 'filename' in elem.tag:
                img_info['filename'] = os.path.join(img_dir, elem.text)
            if 'width' in elem.tag:
                img_info['width'] = int(elem.text)
                assert img_info['width'] == 512
            if 'height' in elem.tag:
                img_info['height'] = int(elem.text)
                assert img_info['height'] == 512
            if 'object' in elem.tag or 'part' in elem.tag:
                # x1-y1-x2-y2-label
                object_info = [0, 0, 0, 0, 0]
                boxes_counter += 1
                for attr in list(elem):
                    if 'name' in attr.tag:
                        label = labels.index(attr.text) +1
                        object_info[4] = label
                    if 'bndbox' in attr.tag:
                        for pos in list(attr):
                            if 'xmin' in pos.tag:
                                object_info[0] = int(pos.text)
                            if 'ymin' in pos.tag:
                                object_info[1] = int(pos.text)
                            if 'xmax' in pos.tag:
                                object_info[2] = int(pos.text)
                            if 'ymax' in pos.tag:
                                object_info[3] = int(pos.text)
                img_info['object'].append(object_info)

        imgs_info.append(img_info) #filename, width, height, box_info
        if boxes_counter > max_boxes:
            max_boxes = boxes_counter

    #[b, 40, 5]
    print(max_boxes)
    boxes = np.zeros([len(imgs_info), max_boxes, 5])
    print(boxes.shape)
    imgs = [] # filename list
    for i, img_info in enumerate(imgs_info):
        #[N, 5]
        img_boxes = np.array(img_info['object'])
        #overwrite the N boxes info
        boxes[i, :img_boxes.shape[0]] = img_boxes 
        imgs.append(img_info['filename'])

        # print(img_info['filename'], boxes[i, :5])
    # imgs: list of image path
    # boxes: [b, 40, 5]
    return imgs, boxes

obj_names = ('sugarbeet', 'weed')
imgs, boxes = parse_annotation('data/train/image', 'data/train/annotation', obj_names)

print(len(imgs))
print(boxes.shape)
# %%

def preprocess(img, img_boxes):
    # img: string
    # img_boxes: [40, 5]
    x = tf.io.read_file(img)
    x = tf.image.decode_png(x, channels=3)
    x = tf.image.convert_image_dtype(x, tf.float32)

    return x, img_boxes


def get_dataset(img_dir, ann_dir, batchsz):
    imgs, boxes = parse_annotation(img_dir, ann_dir, obj_names)
    db = tf.data.Dataset.from_tensor_slices((imgs, boxes))
    db = db.shuffle(1000).map(preprocess).batch(batchsz).repeat()
    print('db images: ', len(imgs))

    return db
# %%
train_db = get_dataset('data/train/image', 'data/train/annotation', 4)
print(train_db)
# %%
