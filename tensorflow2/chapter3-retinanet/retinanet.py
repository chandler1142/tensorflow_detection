data_dir = 'data/baijiu'
result_model_path = 'models/'
ratio = 0.1

import tensorflow as tf

"""
划分数据集
"""
# 开始运行程序
# 数据准备
# data_prep_command_line = ['python', 'data_prep\\split_data.py', '--data_dir', data_dir, '--ratio', str(ratio), ]
# print('\n开始分离训练集和验证集')
# process = subprocess.Popen(data_prep_command_line)
# process.wait()
# if process.returncode != 0:
#     print(f"分离训练集和验证集失败，失败码：{process.returncode}")
# else:
#     print('分离训练集和验证集成功')

"""
生成CSV生成CSV格式数据集和标注
"""
# 生成 csv
# gen_csv_command_line = ['python', 'data_prep/gen_csv.py', '--data_dir', data_dir]
# print('\n开始生成 csv 文件')
# process = subprocess.Popen(gen_csv_command_line)
# process.wait()
# if process.returncode != 0:
#     print(f"生成 csv 文件失败，失败码：{process.returncode}")
# else:
#     print('生成 csv 文件成功')


"""
训练模型
"""
"""
直接运行keras-retinanet
参数：
--batch-size
1
--tensorboard-dir
D:\记录\retinanet_wine
--snapshot-path
D:\workspace\tensorflow_detection\tensorflow2\chapter3-retinanet\models
csv
D:\workspace\tensorflow_detection\tensorflow2\chapter3-retinanet\data\train_data.csv
D:\workspace\tensorflow_detection\tensorflow2\chapter3-retinanet\data\class.csv
--val-annotations
D:\workspace\tensorflow_detection\tensorflow2\chapter3-retinanet\data\val_data.csv
"""


"""
convert to inference file
直接运行convert_model.py
参数
--no-class-specific-filter
D:\workspace\tensorflow_detection\tensorflow2\chapter3-retinanet\models\resnet50_csv_02.h5
D:\workspace\tensorflow_detection\tensorflow2\chapter3-retinanet\models\retinanet_inference.h5
"""

# from keras_retinanet import models
# model = models.load_model('models/resnet50_csv_02.h5',  backbone_name='resnet50')
from keras_retinanet.models import resnet
import keras_resnet
import keras_retinanet
# keras_retinanet.models.load_model('models/resnet50_csv_02.h5', )


# model = tf.keras.models.load_model('models/resnet50_csv_02.h5')
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
# open("converted_model.tflite", "wb").write(tflite_model)


"""
测试单张图
"""
import pandas as pd

df = pd.read_csv("data/class.csv", header=None)

# load label to names mapping for visualization purposes
labels_to_names = df[0].values.tolist()
print(labels_to_names)


model = keras_retinanet.models.load_model("models/retinanet_inference.h5", backbone_name='resnet50')

from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

# load image
image = read_image_bgr('test_1.jpg')

# copy to draw on
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image)

# process image
start = time.time()
boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
end = time.time()
print("processing time: ", end - start)

boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
print("processing time2: ", time.time() - end)

# correct for image scale
boxes /= scale

# visualize detections
for box, score, label in zip(boxes[0], scores[0], labels[0]):
    # scores are sorted so we can break
    if score < 0.5:
        break

    color = label_color(label)

    b = box.astype(int)
    draw_box(draw, b, color=color)
    caption = "{} {:.3f}".format(labels_to_names[label], score)
    draw_caption(draw, b, caption)

plt.figure(figsize=(15, 15))
plt.axis('off')
plt.imshow(draw)
plt.show()

# 保存结果图
cv2.imwrite("test_1_result.jpg", draw)




"""
检测抠小图
"""
# load image
image = read_image_bgr('test_1.jpg')

# copy to draw on
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image)

# process image
start = time.time()
boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
print("processing time: ", time.time() - start)

# correct for image scale
boxes /= scale

idx = 0
# visualize detections
for box, score, label in zip(boxes[0], scores[0], labels[0]):
    # scores are sorted so we can break
    if score < 0.5:
        break

    b = box.astype(int)
    cropped = draw[b[1]:b[3], b[0]:b[2]]
    if cropped is not None:
        cv2.imwrite(f"data/classified_raw/{idx}.jpg", cropped)
    idx+=1