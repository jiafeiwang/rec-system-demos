# -*- coding:utf-8 -*-
# @Author   : Wang Jiahui
# @File     : main.py
import os
from models.DeepFM import DeepFM
from utils.data_utils import create_demo_dataset
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam

if __name__ == '__main__':

    file_path = 'data/demo_data.txt'
    (dense_feature_info, sparse_feature_info), data = create_demo_dataset(file_path)

    model = DeepFM(6, 0.01, 0.01, [32, 16], dense_feature_info, sparse_feature_info)
    optimizer = RMSprop(0.005)
    batch_size, epochs = 32, 30
    summary_writer_dir = os.path.join(os.getcwd(), 'output', 'tensorboard', 'callback')
    checkpoint_path = os.path.join(os.getcwd(), 'output', 'model_files', 'checkpoint', 'model.ckpt')

    X_train, y_train, X_val, y_val = data

    # 定义checkpoint和tensorboard的回调函数
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     save_best_only=True
                                                     )
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=summary_writer_dir)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['AUC', 'accuracy'])

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_data=(X_val, y_val), validation_freq=1,
              callbacks=[cp_callback, tb_callback])

    model.summary()
