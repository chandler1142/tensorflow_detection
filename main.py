# %%
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
from tensorflow import keras
import tensorflow as tf
import numpy as np
import os
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


tf.random.set_seed(2234)
np.random.seed(2234)

# %%
print(tf.__version__)
# print(tf.config.list_physical_devices('GPU'))
print(tf.config.list_logical_devices())

# %%

#1.1 data import
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
                        label = labels.index(attr.text) + 1
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

        imgs_info.append(img_info)  # filename, width, height, box_info
        if boxes_counter > max_boxes:
            max_boxes = boxes_counter

    #[b, 40, 5]
    print(max_boxes)
    boxes = np.zeros([len(imgs_info), max_boxes, 5])
    print(boxes.shape)
    imgs = []  # filename list
    for i, img_info in enumerate(imgs_info):
        #[N, 5]
        img_boxes = np.array(img_info['object'])
        # overwrite the N boxes info
        boxes[i, :img_boxes.shape[0]] = img_boxes
        imgs.append(img_info['filename'])

        # print(img_info['filename'], boxes[i, :5])
    # imgs: list of image path
    # boxes: [b, 40, 5]
    return imgs, boxes


obj_names = ('sugarbeet', 'weed')
imgs, boxes = parse_annotation(
    'data/train/image', 'data/train/annotation', obj_names)

print(len(imgs))
print(boxes.shape)
# %%

#1.2 data preprocess
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
#1.3 dataset 
train_db = get_dataset('data/train/image', 'data/train/annotation', 4)
print(train_db)


# %%
from matplotlib import pyplot as plt
from matplotlib import patches


def db_visualize(db):
    # imgs: [b, 512, 512, 3]
    # imgs_boxes: [b, 40, 5]
    imgs, imgs_boxes = next(iter(db))
    img, img_box = imgs[0], imgs_boxes[0]
    # 创建一张图
    f, ax1 = plt.subplots(1)
    ax1.imshow(img)
    for x1, y1, x2, y2, l in img_box:  # [40, 5]
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        w = x2-x1
        h = y2-y1
        if l ==1: #green for sugarweet
            color = (0,1,0)
        elif l ==2: #red for weed
            color = (1,0,0)
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor=color, facecolor='none')
        ax1.add_patch(rect)
        


# %%
db_visualize(train_db)

# %%
# 1.4 data augmentation
import imgaug as ia
from    imgaug import augmenters as iaa
def augmentation_generator(yolo_dataset):
    '''
    Augmented batch generator from a yolo dataset

    Parameters
    ----------
    - YOLO dataset
    
    Returns
    -------
    - augmented batch : tensor (shape : batch_size, IMAGE_W, IMAGE_H, 3)
        batch : tupple(images, annotations)
        batch[0] : images : tensor (shape : batch_size, IMAGE_W, IMAGE_H, 3)
        batch[1] : annotations : tensor (shape : batch_size, max annot, 5)
    '''
    for batch in yolo_dataset:
        # conversion tensor->numpy
        img = batch[0].numpy()
        boxes = batch[1].numpy()
        # conversion bbox numpy->ia object
        ia_boxes = []
        for i in range(img.shape[0]):
            ia_bbs = [ia.BoundingBox(x1=bb[0],
                                       y1=bb[1],
                                       x2=bb[2],
                                       y2=bb[3]) for bb in boxes[i]
                      if (bb[0] + bb[1] +bb[2] + bb[3] > 0)]
            ia_boxes.append(ia.BoundingBoxesOnImage(ia_bbs, shape=(512, 512)))
        # data augmentation
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Multiply((0.4, 1.6)), # change brightness
            #iaa.ContrastNormalization((0.5, 1.5)),
            #iaa.Affine(translate_px={"x": (-100,100), "y": (-100,100)}, scale=(0.7, 1.30))
            ])
        #seq = iaa.Sequential([])
        seq_det = seq.to_deterministic()
        img_aug = seq_det.augment_images(img)
        img_aug = np.clip(img_aug, 0, 1)
        boxes_aug = seq_det.augment_bounding_boxes(ia_boxes)
        # conversion ia object -> bbox numpy
        for i in range(img.shape[0]):
            boxes_aug[i] = boxes_aug[i].remove_out_of_image().clip_out_of_image()
            for j, bb in enumerate(boxes_aug[i].bounding_boxes):
                boxes[i,j,0] = bb.x1
                boxes[i,j,1] = bb.y1
                boxes[i,j,2] = bb.x2
                boxes[i,j,3] = bb.y2
        # conversion numpy->tensor
        batch = (tf.convert_to_tensor(img_aug), tf.convert_to_tensor(boxes))
        #batch = (img_aug, boxes)
        yield batch
#%%
aug_train_db = augmentation_generator(train_db)
db_visualize(aug_train_db)

# %%
