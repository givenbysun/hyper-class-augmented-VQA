import json
import constants as C
import deepdish as dd

with open(C.data_dir + C.img_features_file, 'r') as json_file:
    image_data = json.load(json_file)
    dd.io.save('{0}{1}'.format(C.data_dir, C.img_features_h5_file), image_data)
