
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.vgg16 import preprocess_input
import numpy as np
import os
# import deepdish as dd
import json
import h5py

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# def get_image_files(dir):
#     images = []
#     for root, dirs, files in os.walk(dir):
#         for filename in files:
#             if '.jpg' in filename:
#                 images.append(filename)
#     print images[0]
#     return images

def get_image_files_by_phase(data, phase=''):
    images = []
    for image_file_name in data.keys():
        images.append(image_file_name)

    # images.sort(key=str.lower)
    return images


def get_model():
    model = VGG16(weights='imagenet', include_top=False)
    model.summary()
    # model.layers.pop()
    # model.layers.pop()
    return model

def preprocess_images(image_dir, images):
    preprocessed_images = {}
    counter = 0
    for img_file in images:
        img_path = image_dir + img_file
        img = image.load_img(img_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        preprocessed_images[img_file] = img_data
        counter = counter +1
        if counter %1000 == 0:
            print 'preprocessed {0} images ... '.format(counter)
            # break
    return preprocessed_images

def extract_image_features(model, preprocessed_images):
    counter = 0
    extracted_image_features = {}
    for img_file in preprocessed_images.keys():
        vgg16_feature = model.predict(preprocessed_images[img_file])
        vgg16_feature = vgg16_feature.reshape((49, 512))
        # vgg16_feature = vgg16_feature.reshape((1000))
        # np.around(vgg16_feature, decimals=6)
        vgg16_feature_list =  np.squeeze(vgg16_feature).tolist()
        extracted_image_features[img_file] = vgg16_feature_list
        counter = counter + 1
        if counter %1000 == 0:
            print 'extracted features from {0} images ... '.format(counter)
            # break

    return extracted_image_features

def generate_vgg_features(extracted_image_features, data, phase):
    counter = 0
    X_image = []
    print '{0} size is: {1}'.format(phase, len(data))
    for image_file_name in extracted_image_features.keys():
        X_image.append(extracted_image_features[image_file_name])
        counter = counter + 1
        if counter % 1000 == 0:
            print counter


    return X_image


def get_data(model, image_dir, images, data, phase):


    extracted_image_features = extract_image_features(model,
                                                      preprocess_images(image_dir, images ))
    print len(extracted_image_features)
    X_image = generate_vgg_features(extracted_image_features, data, phase)

    image_to_index = {}
    index = 0
    for image_file_name in images:
        image_to_index[image_file_name] = index
        index = index + 1

    return X_image, image_to_index

def perform():
    image_dir = "/home/mehrdad/PycharmProjects/imSitu/resized_256/of500_images_resized/"
    data_dir = "/home/mehrdad/PycharmProjects/imSitu_vqa_baseline/data/"
    model = get_model()

    train = json.load(open('{0}/train.json'.format(data_dir)))
    dev = json.load(open('{0}/dev.json'.format(data_dir)))
    test = json.load(open('{0}/test.json'.format(data_dir)))

    X_image = {}
    image_to_index = {}

    counter = 0
    images = get_image_files_by_phase(train)

    partitions = {}
    for i in range(len(images)):
        k = int(i/1000)
        if k not in partitions.keys():
            partitions[k] = []
        partitions[k].append(i)

#    for k in range(len(partitions.keys())):
#
#       images_filtered = [ images[i] for i in partitions[k]]

#        [X_image['train'], image_to_index['train']] = get_data(model, image_dir, images_filtered, train, 'train')

#        index = (5-len(str(k)))*'0' + str(k+1)

#        h5f = h5py.File(data_dir + 'imSitu_vgg_49_512_features_{0}/imSitu_vgg_49_512_features_{0}_{1}.h5'.format('train', index), 'w')
#        h5f.create_dataset('train', data=X_image['train'])
#        h5f.close()

#        counter = counter + 1
#        if counter % 1000 == 0:
#            print counter
    index = 0
    image_to_index['train'] = {}
    for img in images:
        image_to_index['train'][img] = index
        index = index + 1

    with open('{0}{1}'.format(data_dir, 'image_to_index_train.json'), 'w') as fp:
        json.dump(image_to_index['train'], fp)






    counter = 0
    images = get_image_files_by_phase(test)

    partitions = {}
    for i in range(len(images)):
        k = int(i / 1000)
        if k not in partitions.keys():
            partitions[k] = []
        partitions[k].append(i)

#    for k in range(len(partitions.keys())):

#       images_filtered = [images[i] for i in partitions[k]]

#        [X_image['test'], image_to_index['test']] = get_data(model, image_dir, images_filtered, test, 'test')

#        index = (5 - len(str(k))) * '0' + str(k + 1)

#        h5f = h5py.File(
#            data_dir + 'imSitu_vgg_49_512_features_{0}/imSitu_vgg_49_512_features_{0}_{1}.h5'.format('test', index),
#            'w')
#        h5f.create_dataset('test', data=X_image['test'])
#        h5f.close()

#        counter = counter + 1
#        if counter % 1000 == 0:
#            print counter

    index = 0
    image_to_index['test'] = {}
    for img in images:
        image_to_index['test'][img] = index
        index = index + 1


    with open('{0}{1}'.format(data_dir, 'image_to_index_test.json'), 'w') as fp:
        json.dump(image_to_index['test'], fp)





    # [X_image['test'], image_to_index['test']] = get_data(model, image_dir,  test, 'test')
    #
    # h5f = h5py.File(data_dir + 'imSitu_vgg_49_512_features_test.h5'.format('test'), 'w')
    # h5f.create_dataset('test', data=X_image['test'])
    # h5f.close()
    #
    # with open('{0}{1}'.format(data_dir, 'image_to_index_test.json'), 'w') as fp:
    #     json.dump(image_to_index['test'], fp)

if __name__ == '__main__':
    perform()

