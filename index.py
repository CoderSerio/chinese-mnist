from pathlib import Path
import pandas as pd
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


# 数据读取&预处理
def load_data():
    images = []
    labels = []
    df = pd.read_csv('./dataset/chinese_mnist.csv')
    # 数据标注的形式是，csv里存的是文件名的后缀数字，映射到正确的答案
    # 图片已经是64x64的了
    folder_path = './dataset/data/data'
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        image = Image.open(image_path).convert('RGB')

        # 归一化
        image_nd_array = np.array(image) / 255
        images.append(image_nd_array.flatten())
        # 先去掉扩展名，
        image_ids = ''.join(image_name.split('.')[0]).split('_')[1:]
        # suite_id,sample_id,code
        target_df = df[
            (df['suite_id'] == int(image_ids[0])) &
            (df['sample_id'] == int(image_ids[1])) &
            (df['code'] == int(image_ids[2]))
        ]
        label = target_df['value'].values[0]
        if label <= 10:
            labels.append(label)
        elif label == 100:
            labels.append(11)
        elif label == 1000:
            labels.append(12)
        elif label == 10000:
            labels.append(13)
        elif label == 100000000:
            labels.append(14)

    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels),  test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test


def train(images, labels):
    print('images.shape \n',  images.shape)
    print('labels: \n',  labels)
    model = tf.keras.Sequential()
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    # 有十五个类别，包括0-10，百千万亿
    model.add(layers.Dense(15, activation='softmax'))

    # 设置优化器、损失函数、评估函数
    # labels是 one_hat, 且类型是多分类，用稀疏交叉熵SparseCategoricalCrossentropy作为损失函数就可以了
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(images, labels, epochs=10)
    model.save('./model/chinese_mnist.keras')
    return model


def predict(model, x_test, y_test):
    y_predicted = model.predict(x_test)
    y_predicted_classes = np.argmax(y_predicted, axis=1)
    accuracy = np.sum(y_predicted_classes == y_test) / len(y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return y_predicted


x_train, x_test, y_train, y_test = load_data()
model = train(x_train, y_train)
prediction_res = predict(model, x_test, y_test)
