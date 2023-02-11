# -*- coding:utf-8 -*-
# @Author   : Wang Jiahui
# @File     : graph_train.py

import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras.metrics import AUC, Mean, BinaryAccuracy
import time
import os
from models.DeepFM import DeepFM
from utils.data_utils import create_demo_dataset
from tensorflow.keras.optimizers import RMSprop

# 静态图加速
@tf.function
def train_one_step(X_train_batch,y_train_batch):
    with tf.GradientTape() as tape:
        y_pred = model(X_train_batch)

        # 计算基础损失，其中y_train_tatch [None,1] y_pred [None, 1]
        loss = tf.reduce_mean(losses.binary_crossentropy(y_train_batch,y_pred))

        # 加入模型的正则损失
        loss += tf.reduce_sum(model.losses)

    grads = tape.gradient(loss, model.trainable_variables)

    # 更新model系数和optimizer对应的学习参数
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
    # 训练集模型评价指标计算
    train_loss.update_state(loss)
    train_acc.update_state(y_train_batch, y_pred)
    train_auc.update_state(y_train_batch, y_pred)

@tf.function
def valid_one_step(X_val_batch, y_val_batch):
    pred = model(X_val_batch)
    loss = losses.binary_crossentropy(y_val_batch, pred)
    loss = tf.reduce_mean(loss)

    # 验证集模型评价指标计算
    val_loss.update_state(loss)
    val_acc.update_state(y_val_batch, pred)
    val_auc.update_state(y_val_batch, pred)

if __name__ == '__main__':

    file_path = 'data/demo_data.txt'
    (dense_feature_info, sparse_feature_info), data = create_demo_dataset(file_path)

    model = DeepFM(6, 0.01, 0.01, [32, 16], dense_feature_info, sparse_feature_info)
    optimizer = RMSprop(0.005)
    batch_size, epochs = 32, 30
    summary_writer_dir = os.path.join(os.getcwd(), 'output', 'tensorboard')
    model_dir = os.path.join(os.getcwd(), 'output', 'model_files', 'base')

    X_train, y_train, X_val, y_val = data

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

    model_path = os.path.join(model_dir, f'base_{int(time.time())}.h5')
    model_path_last = os.path.join(model_dir, f'base_last.h5')

    # 模型训练评估指标
    train_loss = Mean(name='train_loss')
    train_acc = BinaryAccuracy(name='train_acc')
    train_auc = AUC(name='train_auc')

    # 模型验证评估指标
    val_loss = Mean(name='val_loss')
    val_acc = BinaryAccuracy(name='val_acc')
    val_auc = AUC(name='val_auc')

    summary_writer_train = tf.summary.create_file_writer(
        os.path.join(summary_writer_dir, 'base', 'train')
    )

    summary_writer_val = tf.summary.create_file_writer(
        os.path.join(summary_writer_dir, 'base', 'val')
    )

    for epoch in range(epochs):
        start = time.time()
        print(f'Epoch {epoch + 1} / {epochs}')

        # 训练
        for batch, (X_train_batch, y_train_batch) in enumerate(train_dataset, 1):
            train_one_step(X_train_batch, y_train_batch)

        # 验证
        for batch, (X_val_batch, y_val_batch) in enumerate(val_dataset):
            valid_one_step(X_val_batch, y_val_batch)

        epoch_run_time = int((time.time() - start) % 60)
        print(
            'ETA : %ss, loss : %s, accuracy: %s, auc: %s || val_loss : %s, val_accuracy: %s, val_auc: %s'
            % (
                epoch_run_time,
                train_loss.result().numpy(), train_acc.result().numpy(), train_auc.result().numpy(),
                val_loss.result().numpy(), val_acc.result().numpy(), val_auc.result().numpy()
            ))

        # 记录指标到tensorboard日志
        with summary_writer_train.as_default():
            tf.summary.scalar("loss", train_loss.result().numpy(), step=epoch)
            tf.summary.scalar("auc", train_auc.result().numpy(), step=epoch)

        with summary_writer_val.as_default():
            tf.summary.scalar("loss", val_loss.result().numpy(), step=epoch)
            tf.summary.scalar("auc", val_auc.result().numpy(), step=epoch)

        # 重置指标
        train_loss.reset_states()
        train_acc.reset_states()
        train_auc.reset_states()

        val_loss.reset_states()
        val_acc.reset_states()
        val_auc.reset_states()

    # 保存模型
    model.save_weights(model_path)
    model.save_weights(model_path_last)
