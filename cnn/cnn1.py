import numpy as np
import tensorflow as tf
import dataset.cfar10 as ds
from matplotlib import pyplot as plt
import os.path

# CNN to check if an image is of automobile class or not

MODEL_FILE = '/tmp/cnn1.tf'


def features_to__image(features):
    return features.reshape(3, 32, 32).transpose(1, 2, 0).astype('uint8')


def setup_cnn(mode, dataset, shape):
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

    logits = tf.layers.dense(inputs=dropout, units=2, name='logits')
    print("logits: ", logits.shape)

    classes = tf.argmax(logits, axis=1)
    probability = tf.nn.softmax(logits, name='probability')

    merged = tf.summary.merge_all()

    return logits, classes, probability, input_layer


dataset = ds.get_dataset()

features = dataset['data']
labels = [1 if x == 1 or x == 9 else 0 for x in dataset['labels']]

# for i, x in enumerate(labels):
#     if x == 1:
#         pic = features[i].reshape(3, 32, 32).transpose(1, 2, 0).astype('uint8')
#         plt.imshow(pic, interpolation='nearest', aspect='equal')
#         plt.show()

test0 = features_to__image(features[3])
# plt.figure(num=None, figsize=(1, 1))
# plt.imshow(test0)
# plt.show()

graph = tf.Graph()
with graph.as_default():
    logits, classes, probability, inputs = setup_cnn(tf.estimator.ModeKeys.PREDICT,
                                                     np.array(features, dtype=np.float32), [3, 32, 32])

with tf.Session(graph=graph) as session:
    saver = tf.train.Saver()
    # train_writer = tf.summary.FileWriter('/tmp/test1/train', session.graph)
    # test_writer = tf.summary.FileWriter('/tmp/test1/test')

    if os.path.isfile(MODEL_FILE+".index"):
        saver.restore(session, MODEL_FILE)
    else:
        tf.global_variables_initializer().run()
        print('saving model to file')
        saver.save(session, MODEL_FILE)

    l, c, p = session.run([logits, classes, probability], {inputs: [test0]})
    print ("logits: %s, classes: %s, prediction: %s" % (l, c, p))
