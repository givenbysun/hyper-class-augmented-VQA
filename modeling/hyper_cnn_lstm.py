import numpy as np

from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import ModelCheckpoint
import json
import h5py
import os
import argparse
import codecs
import deepdish as dd
import constants as C
import tensorflow as tf
from tensorflow.python.keras import backend as K_backend
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 8.0
K_backend.set_session(tf.Session(config=config))

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def read_data(data_limit, type='train'):
    print("Reading Data...")
    # reader = codecs.getreader("utf-8")
    # with open('vqa_train_1000.json', 'rb') as fp:
    # with open('vqa_{0}.json'.format(type), 'rb') as fp:
    #     IQA_dataset = json.load(reader(fp))
    #     img_data = IQA_dataset['X'][0]
    #     ques_data = IQA_dataset['X'][1]
    #     ans_data = IQA_dataset['Y']

    # IQA_dataset = dd.io.load(C.data_dir + 'vqa_{0}_c.h5'.format(type))
    IQA_dataset = dd.io.load(C.data_dir + 'vqa_{0}_info.h5'.format(type))
    img_data = IQA_dataset['X'][0]
    ques_data = IQA_dataset['X'][1]
    ans_data = IQA_dataset['Y']
    role_data = IQA_dataset['Yrole']

    print("... Done")
    # index =0
    # for q in ques_data:
    #     for item in q:
    #         if item>=4417:
    #             print(index , q)
    #     index = index + 1

    img_data = np.array(img_data)

    # Normalizing images
    tem = np.sqrt(np.sum(np.multiply(img_data, img_data), axis=1))
    train_img_data = np.divide(img_data, np.transpose(np.tile(tem ,(4096 ,1))))

    ques_train = np.array(ques_data)

    train_X = [train_img_data, ques_train]

    # train_y = []
    # for samples in ans_data:
    #     if sum(samples)==0:
    #         y=[]
    #         for i in range(999):
    #             y.append(0)
    #         y.append(1)
    #         train_y.append(y)
    #     else:
    #         train_y.append(samples)
    #     index = 0
    #     for i in samples:
    #         if i==1:
    #             print index
    #         index = index + 1


    train_y = np.array(ans_data)
    train_y_r = np.array(role_data)

    print 'SHAPE is {0}'.format(train_y_r.shape)

    return train_X, train_y, train_y_r, None


def get_metadata():
    meta_data = json.load(open(C.data_dir + C.vocabulary_file, 'r'))
    meta_data['ix_to_word'] = {word: int(i) for i, word in meta_data['ix_to_word'].items()}
    meta_data['ix_to_role'] = {role: int(i) for i, role in meta_data['ix_to_role'].items()}
    return meta_data

def prepare_embeddings(num_words, embedding_dim, metadata):
    # embedding_matrix_filename = ''
    # if os.path.exists(embedding_matrix_filename):
    #     with h5py.File(embedding_matrix_filename) as f:
    #         return np.array(f['embedding_matrix'])

    print("Embedding Data...")
    # with open(train_questions_path, 'r') as qs_file:
    #     questions = json.loads(qs_file.read())
    #     texts = [str(_['question']) for _ in questions['questions']]

    embeddings_index = {}
    with open(C.data_dir + C.glove_path, 'r') as glove_file:
        for line in glove_file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((num_words, embedding_dim))
    word_index = metadata['ix_to_word']

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        # print word, i
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # with h5py.File(embedding_matrix_filename, 'w') as f:
    #     f.create_dataset('embedding_matrix', data=embedding_matrix)

    return embedding_matrix

num_words = 20000
embedding_dim = 300
seq_length =    26
ckpt_model_weights_filename =  C.data_dir + C.experiment_multi + C.ckpt_model_weights_file
model_weights_filename = C.data_dir+ C.experiment_multi + C.model_weights_file

predictions_file_name = C.data_dir + C.experiment_multi + C.predictions_file_name
predictions_with_roles_file_name = C.data_dir + C.experiment_multi + 'ROLE' + C.predictions_file_name
answers_file_name = C.data_dir + C.experiment_multi + C.answers_file_name

from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Input, Dense, Activation, Dropout, LSTM, Flatten, Embedding, Multiply
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
import h5py
from tensorflow.python.keras.utils import multi_gpu_model

def Word2VecModel(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate):
    print("Creating text model...")
    model = Sequential()
    model.add(Embedding(num_words, embedding_dim,
        weights=[embedding_matrix], input_length=seq_length, trainable=False))
    model.add(LSTM(units=512, return_sequences=True, input_shape=(seq_length, embedding_dim)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=512, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1024, activation='tanh'))
    return model

def img_model(dropout_rate):
    print("Creating image model...")
    model = Sequential()
    model.add(Dense(1024, input_dim=4096, activation='tanh'))
    return model

def vqa_model(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes, num_classes_r):
    vgg_model = img_model(dropout_rate)
    lstm_model = Word2VecModel(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate)
    print("Merging final model...")

    # fc_model.add(Merge([vgg_model, lstm_model], mode='mul'))

    iL = [Input(shape=(4096,)), Input(shape=(seq_length,))]
    # iL = [Input(shape=vgg_model.input.shape), Input(shape=lstm_model.input.shape)]
    hL = [vgg_model(iL[0]), lstm_model(iL[1])]
    oL = Multiply()(hL)
    merge_model = Model(inputs=iL, outputs=oL)
    # merge_model = Sequential(inputs=iL, outputs=oL)

    fc_model = Sequential()
    # fc_model.add(Dropout(dropout_rate))
    fc_model.add(Dense(1000, input_shape=(1024,), activation='tanh'))
    fc_model.add(Dropout(dropout_rate))

    composed_model = Model(
        inputs=merge_model.input,
        outputs=fc_model(merge_model.output)
    )

    fc_temp_1 = Sequential(name = 'answer_output')
    fc_temp_1.add(Dense(num_classes, input_shape=(1000,), activation='softmax'))
    fc_model_1 = Model(
        inputs=composed_model.input,
        outputs=fc_temp_1(composed_model.output),
    )

    fc_temp_2 = Sequential(name = 'role_output')
    fc_temp_2.add(Dense(num_classes_r, input_shape=(1000,), activation='softmax'))
    fc_model_2 = Model(
        inputs=composed_model.input,
        outputs=fc_temp_2(composed_model.output),
    )

    # fc_model_1.add(Dense(num_classes, activation='softmax'))
    # fc_model_2.add(Dense(num_classes_r, activation='softmax'))


    final_model = Model(
        inputs=merge_model.input,
        outputs=[fc_model_1.output, fc_model_2.output]
    )

    losses = {
        'answer_output': 'categorical_crossentropy',
        'role_output': 'categorical_crossentropy',
    }

    lossWeights = {'answer_output': 1.0, 'role_output': 1.0}

    # composed_model = multi_gpu_model(composed_model, gpus=3)
    final_model.compile(optimizer='rmsprop', loss=losses, loss_weights = lossWeights,
                           metrics=['accuracy'])
    return final_model



def get_model(dropout_rate, model_weights_filename):
    print("Creating Model...")
    metadata = get_metadata()
    num_classes = len(metadata['ix_to_ans'].keys())
    num_roles = len(metadata['ix_to_role'].keys())
    num_words = len(metadata['ix_to_word'].keys())

    embedding_matrix = prepare_embeddings(num_words, embedding_dim, metadata)
    model = vqa_model(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes, num_roles)
    if os.path.exists(model_weights_filename):
        print("Loading Weights...")
        model.load_weights(model_weights_filename)

    return model

def train(args):
    dropout_rate = 0.5
    train_X, train_y, train_y_r, qids = read_data(args.data_limit)
    X_test, y_test, y_r_test, ignore = read_data(args.data_limit, 'test')
    model = get_model(dropout_rate, model_weights_filename)
    checkpointer = ModelCheckpoint(filepath=ckpt_model_weights_filename,verbose=1)
    # print('train_X is of size ')
    # print(len(train_X))
    # model.fit(train_X, train_y, epochs=args.epoch, batch_size=args.batch_size, callbacks=[checkpointer], shuffle="batch")
    model.fit(train_X, {"answer_output": train_y, "role_output": train_y_r},
              validation_data=(X_test,{"answer_output": y_test, "role_output": y_r_test}), epochs=args.epoch, batch_size=args.batch_size, callbacks=[checkpointer], verbose=2)
    model.save_weights(model_weights_filename, overwrite=True)

def val(args):
    val_X, val_y, val_y_r, ignore = read_data(args.data_limit, 'test')
    model = get_model(0.0, model_weights_filename)
    print("Evaluating Accuracy on validation set:")
    metric_vals = model.evaluate(val_X, {"answer_output": val_y, "role_output": val_y_r}, batch_size=5000)
    print("")
    for metric_name, metric_val in zip(model.metrics_names, metric_vals):
        print(metric_name, " is ", metric_val)

    pred_y = model.predict(val_X, batch_size=5000)[0] # answers only
    pred_y_r = model.predict(val_X, batch_size=5000)[1] # answers only

    gold_answers = []
    gold_roles = []
    predicted_answers = []
    predicted_roles = []
    vocab_dict = get_metadata()
    vocab_dict['ix_to_role'] = {y: x for x, y in vocab_dict['ix_to_role'].iteritems()}
    # yg = val_y.tolist()
    # yp = pred_y.tolist()
    N = len(pred_y)
    for i in range(N):
        # score = sum(e[0] * e[1] for e in zip(yg[i], yp[i]))
        # if score > 0:
        #     count_corrects = count_corrects + 1

        predicted = np.argmax(pred_y[i])
        predicted_answers.append(vocab_dict['ix_to_ans'][str(predicted)])

        answer = np.argmax(val_y[i])
        gold_answers.append(vocab_dict['ix_to_ans'][str(answer)])

        role = np.argmax(pred_y_r[i])
        predicted_roles.append(vocab_dict['ix_to_role'][role])

        role = np.argmax(val_y_r[i])
        gold_roles.append(vocab_dict['ix_to_role'][role])

    # f = open(predictions_file_name ,'w')
    f = open(predictions_with_roles_file_name ,'w')
    f.write('actual Answer, actual Role, predicted Answer, predicted Role\n')
    for i in range(len(gold_answers)):
        f.write('{0}, {1}, {2}, {3}\n'.format(gold_answers[i], gold_roles[i], predicted_answers[i],predicted_roles[i]))
    f.close()

    # print 'MANUAL ACCURACY IS : {0}'.format(count_corrects/N)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='val')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--data_limit', type=int, default=215359, help='Number of data points to fed for training')
    args = parser.parse_args()

    if args.type == 'train':
        train(args)
    elif args.type == 'val':
        val(args)

def run_code(phase='train'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default=phase)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--data_limit', type=int, default=215359, help='Number of data points to fed for training')
    args = parser.parse_args()

    if args.type == 'train':
        train(args)
    elif args.type == 'test':
        val(args)

