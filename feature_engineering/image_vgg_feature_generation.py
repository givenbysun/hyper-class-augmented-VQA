
import json
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras import backend as K_backend
# from tensorflow.keras.utils import multi_gpu_model
import constants as C
import deepdish as dd


# https://medium.com/@franky07724_57962/using-keras-pre-trained-models-for-feature-extraction-in-image-clustering-a142c6cdf5b1

img_dir = C.img_dir
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"


config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 8.0
K_backend.set_session(tf.Session(config=config))


def get_vgg_model():
    model = VGG16(weights='imagenet', include_top=True)
    model.layers.pop()
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    # model.layers[-1].outbound_nodes = []
    # model = multi_gpu_model(model, gpus=3)
    # model.summary()
    return model


def get_image_representation(model, img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    vgg16_feature = model.predict(img_data)

    # print vgg16_feature.shape
    return np.squeeze(vgg16_feature).tolist()


def generate_vgg_features(data, phase):
    counter = 1
    X_image = {}
    print '{0} size is: {1}'.format(phase, len(data))
    for image_file_name in data.keys():
        X_image[image_file_name] = get_image_representation(VGG_model, img_dir + image_file_name)
        if counter % 1000 == 0:
            print counter
        counter = counter + 1

    return X_image



VGG_model = get_vgg_model()


train = json.load(open('{0}/train.json'.format(C.data_dir)))
dev = json.load(open('{0}/dev.json'.format(C.data_dir)))
test = json.load(open('{0}/test.json'.format(C.data_dir)))

X_image = {}
X_image['train'] = generate_vgg_features(train, 'train')
X_image['dev'] = generate_vgg_features(dev, 'dev')
X_image['test'] = generate_vgg_features(test, 'test')


dd.io.save(C.data_dir + 'imSitu_vgg_features.h5', X_image)

# with open('imSitu_vgg_features.json', 'w') as fp:
#     json.dump(X_image, fp)