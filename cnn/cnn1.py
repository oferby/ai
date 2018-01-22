import numpy as np
import tensorflow as tf

mode = tf.estimator.ModeKeys.TRAIN

def setup_cnn():
    features = np.random.random_sample((784, 1)).astype(dtype=np.float32)
    print("features: ", features.shape)

    input_layer = tf.reshape(features, [-1, 28, 28, 1])
    print("input_layer: ", input_layer.shape)

    conv1 = tf.layers.conv2d(inputs=input_layer,
                             filters=32,
                             kernel_size=[5, 5],
                             padding='same',
                             activation=tf.nn.relu
                             )
    print("conv1: ", conv1.shape)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    print("pool1: ", pool1.shape)

    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=64,
                             kernel_size=[5, 5],
                             padding='same',
                             activation=tf.nn.relu)
    print("conv2: ", conv2.shape)

    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[2, 2],
                                    strides=2)
    print("pool2: ", pool2.shape)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    print("pool2_flat: ", pool2_flat.shape)

    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    print("dense: ", dense.shape)

    dropout = tf.layers.dropout(inputs=dense,
                                rate=0.4,
                                training=mode == tf.estimator.ModeKeys.TRAIN)
    print("dropout: ", dropout.shape)

    logits = tf.layers.dense(inputs=dropout, units=10)
    print("logits: ", logits.shape)
