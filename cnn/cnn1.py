import numpy as np
import tensorflow as tf
import dataset.cfar10 as ds
from matplotlib import pyplot as plt
import os.path

# CNN to check if an image is of automobile class or not

MODEL_FILE = '/tmp/cnn1.tf'


def features_to__image(features):
    return features.reshape(3, 32, 32).transpose(1, 2, 0).astype('uint8')


def setup_cnn(mode):
    # input_layer = tf.reshape(dataset, [-1, shape[0], shape[1], shape[2]])
    input_layer = tf.placeholder(tf.float32, [None, 32, 32, 3], name='input')

    print("input_layer: ", input_layer.shape)

    conv1 = tf.layers.conv2d(inputs=input_layer,
                             filters=32,
                             kernel_size=[5, 5],
                             padding='same',
                             activation=tf.nn.relu,
                             name='conv1'
                             )
    print("conv1: ", conv1.shape)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name='pool1')
    print("pool1: ", pool1.shape)

    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=64,
                             kernel_size=[5, 5],
                             padding='same',
                             activation=tf.nn.relu,
                             name='conv2')
    print("conv2: ", conv2.shape)

    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[2, 2],
                                    strides=2,
                                    name='pool2')
    print("pool2: ", pool2.shape)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64], name='pool2-flat')
    print("pool2_flat: ", pool2_flat.shape)

    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu, name='dense')
    print("dense: ", dense.shape)

    dropout = tf.layers.dropout(inputs=dense,
                                rate=0.4,
                                training=mode == tf.estimator.ModeKeys.TRAIN,
                                name='dropout')
    print("dropout: ", dropout.shape)

    logits = tf.layers.dense(inputs=dropout, units=10, name='logits')
    print("logits: ", logits.shape)

    classes = tf.argmax(logits, axis=1)
    probability = tf.nn.softmax(logits, name='probability')

    merged = tf.summary.merge_all()

    label = tf.placeholder(tf.float32, [None], name='label')

    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(label, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

    tf.summary.scalar('xentropy', cross_entropy_mean)
    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_mean)

    return input_layer, label, train_op, cross_entropy_mean, classes, probability


dataset = ds.get_dataset()

features = dataset['data']
labels = dataset['labels']

# indx = 7
# test0 = features_to__image(features[indx])
# plt.figure(num=None, figsize=(1, 1))
# plt.imshow(test0)
# plt.show()

mode = tf.estimator.ModeKeys.TRAIN

graph = tf.Graph()
with graph.as_default():
    inputs, label, train_op, loss, prediction, probability = setup_cnn(mode)

with tf.Session(graph=graph) as session:
    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter('/tmp/cnn1/train', session.graph)
    test_writer = tf.summary.FileWriter('/tmp/cnn1/test')

    if os.path.isfile(MODEL_FILE + ".index"):
        saver.restore(session, MODEL_FILE)
    else:
        tf.global_variables_initializer().run()

    if mode == tf.estimator.ModeKeys.PREDICT:
        prediction_, probability_ = session.run([prediction, probability], {inputs: [test0]})
        print ("classes: %s, prediction: %s, real: %s" % (prediction_, probability_, labels[indx]))

    elif mode == tf.estimator.ModeKeys.TRAIN:
        for k in range(100):
            i = 1
            mini_batch = 100
            while i + mini_batch < 10000:
                images = []
                labels_batch = []
                for j in range(mini_batch):
                    image = features[j + i - 1].reshape(3, 32, 32).transpose(1, 2, 0).astype('uint8')
                    images.append(image)
                    labels_batch.append(labels[j + i - 1])

                _, loss_ = session.run([train_op, loss], {inputs: images, label: labels_batch})
                i += mini_batch
                print ("loss", loss_)

        print('saving model to file')
        saver.save(session, MODEL_FILE)
