data_dir = 'data/baijiu'
result_model_path = 'models/'
ratio = 0.1

import os
import subprocess

import pandas as pd
from matplotlib import patches
from matplotlib import pyplot as plt

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