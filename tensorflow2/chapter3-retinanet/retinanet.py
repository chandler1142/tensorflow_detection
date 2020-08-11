data_dir = 'data/baijiu'
result_model_path = 'models/'
ratio = 0.1

import subprocess
import os

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
train_csv_path = os.path.join(data_dir, 'train_data.csv')
val_csv_path = os.path.join(data_dir, 'val_data.csv')
class_csv_path = os.path.join(data_dir, 'class.csv')
train_log_path = 'logs/train_info.txt'

if os.path.exists(train_log_path):
    os.remove(train_log_path)

train_log = open(train_log_path, 'a')

train_command_line = ['python', 'keras-retinanet/keras_retinanet/bin/train.py', '--snapshot-path', result_model_path,
                      'csv', train_csv_path, class_csv_path, '--val-annotations', val_csv_path]
print('\n开始训练...')
process = subprocess.Popen(train_command_line, stdout=train_log, stderr=train_log)
process.wait()
if process.returncode != 0:
    print(f"训练失败，失败码：{process.returncode}")
else:
    print('训练完成!')