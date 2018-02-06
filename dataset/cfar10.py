import cPickle

# DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

image_file = '/home/stack/Downloads/cifar-10-batches-py/data_batch_1'
predict_file = '/home/stack/Downloads/cifar-10-batches-py/data_batch_2'


def get_dataset(train=True):
    if train:
        load_file = image_file
    else:
        load_file = predict_file
    with open(load_file, 'rb') as f:
        dict = cPickle.load(f)
    return dict
