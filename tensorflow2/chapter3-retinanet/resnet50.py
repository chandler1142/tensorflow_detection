cate = "classified"
dataset_path = f"data/{cate}"
train_dataset_path = f"{dataset_path}/train"
val_dataset_path = f"{dataset_path}/val"

# """
# 数据划分
# """
# from data_prep.dataset import Dataset
#
# train_ratio = 0.9
# data_loader = Dataset(dataset_path, train_ratio=train_ratio)

"""
列出数据分类
"""
import os

class_name = list(
    filter(
        lambda x: not x.startswith(".")
                  and os.path.isdir(os.path.join(train_dataset_path, x)),
        os.listdir(train_dataset_path),
    )
)
print(f"类别总数：{len(class_name)}")

model = "ResNet50"
model_name = f"{model}_{cate}"
model_path = f"models/{model}/{model_name}.h5"

"""
模型准备
"""
from tensorflow.keras.applications import ResNet50

base_model = ResNet50(include_top=False, weights='imagenet', classes=1000)
base_model.trainable = False

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import Model

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
y = Dense(len(class_name), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=y, name=model_name)

model.summary()

"""
定义优化器和损失函数
"""

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

"""
数据生成器
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_dataset_path,
    target_size=(224, 224),
    classes=class_name,
    color_mode='rgb',
    batch_size=128,
    class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
    val_dataset_path,
    target_size=(224, 224),
    classes=class_name,
    color_mode='rgb',
    batch_size=64,
    class_mode='categorical')

"""
设置 EarlyStoping 条件
"""
from tensorflow.keras.callbacks import EarlyStopping

earlystop = EarlyStopping(monitor="val_loss", patience=3, verbose=1)

"""
开始训练
"""
history = model.fit(
    train_generator,
    epochs=50,
    steps_per_epoch=None,
    validation_data=val_generator,
    validation_steps=5,
    callbacks=earlystop)

"""
模型评估
"""
model.evaluate(
    val_generator,
    steps=10,
    callbacks=earlystop,
    verbose=1)

"""
模型保存
"""
model.save(model_path)

"""
加载模型
"""
import tensorflow as tf

model = tf.keras.models.load_model(model_path)


model = Model(inputs=base_model.input, outputs=y, name=model_name)




"""
单张预测
"""
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
import time


def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    plt.imshow(img_tensor[0])
    plt.axis('off')
    plt.show()

    return img_tensor


img_path = 'data/classified/val/一品苏黄/1031_14884.jpg'
start = time.time()
img_tensor = load_image(img_path)
pred = model.predict(img_tensor)
end = time.time()
print("cost %f" % (end - start))

start = time.time()
img_tensor = load_image(img_path)
pred = model.predict(img_tensor)
end = time.time()
print("cost %f" % (end - start))

start = time.time()
img_tensor = load_image(img_path)
pred = model.predict(img_tensor)
end = time.time()
print("cost %f" % (end - start))

# print([[class_name[i], prob] for i, prob in enumerate(pred[0])])
print([class_name[np.argmax(prob)] for prob in pred])

"""
批量预测
"""


# 加载一批图像
def load_image_list(test_dir):
    image_tensor_list = []
    image_list = [os.path.join(test_dir, img_name) for img_name in os.listdir(test_dir)]
    for image_path in image_list:
        img = image.load_img(image_path, target_size=(224, 224))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        image_tensor_list.append(img_tensor.reshape(224, 224, 3))

    return np.asarray(image_tensor_list)


img_list = load_image_list("data/classified/val/一品苏黄/")
preds = model.predict_on_batch(img_list)
results = [class_name[np.argmax(prob)] for prob in preds]
print(results)
