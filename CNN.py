import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import model_selection
import pkg_resources
import time
import os
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.models import load_model
import matplotlib.pyplot as plt

train = pd.read_csv('./data/train.csv')
train = train[:50]

#Extract all comments as strings
train['comment_text'] = train['comment_text'].astype(str)

identities = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

#Converting target and identities columns into booleans
#(If value >= 0.5, label as toxic, non-toxic otherwise)
def convert_to_bool(df):
    bool_df = df.copy()
    for col in ['target'] + identities:
        bool_df[col] = np.where(df[col] >= 0.5, True, False)
    return bool_df

train = convert_to_bool(train)

#Splitting data into 80% training set and 20% validation set
train_df, validate_df = model_selection.train_test_split(train, test_size=0.2)


MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 200
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(train_df['comment_text'])

#Padding comments to make the vectors of the same length
def pad_text(texts, tokenizer):

    return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_SEQUENCE_LENGTH)


#Training the CNN

EMBEDDINGS_DIM = 50 #Embedding dimension will take values 50, 100, and 200.
PATH = './data/glove.6B.50d.txt'
EPOCHS = 7
L_R = 0.00006
DROPOUT = 0.25
BATCH_SIZE = 128
history = None


def train_model(train_df, validate_df, tokenizer):

    #Preprocessing the training and validation data frames

    text_train = pad_text(train_df['comment_text'], tokenizer)
    labels_train = to_categorical(train_df['target'])

    text_val = pad_text(validate_df['comment_text'], tokenizer)
    labels_val = to_categorical(validate_df['target'])

    print('loading embeddings')
    embeddings_idx = {}
    with open(PATH, encoding='utf-8') as f:
        for line in f:
            vals = line.split()
            word = vals[0]
            coefs = np.asarray(vals[1:], dtype='float32')
            embeddings_idx[word] = coefs

    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, EMBEDDINGS_DIM))
    embedding_words = 0
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_idx.get(word)
        if embedding_vector is not None:
            embedding_words += 1
            embedding_matrix[i] = embedding_vector

    #Creating CNN Layers
    def get_cnn_layers():
        input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedding_layer = Embedding(len(tokenizer.word_index) + 1, EMBEDDINGS_DIM, weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH, trainable=False)
        z = embedding_layer(input)
        z = Conv1D(128, 2, activation='relu', padding='same')(z)
        z = MaxPooling1D(5, padding='same')(z)
        z = Conv1D(128, 3, activation='relu', padding='same')(z)
        z = MaxPooling1D(5, padding='same')(z)
        z = Conv1D(128, 4, activation='relu', padding='same')(z)
        z = MaxPooling1D(40, padding='same')(z)
        z = Flatten()(z)
        z = Dropout(DROPOUT)(z)
        z = Dense(128, activation='relu')(z)
        output = Dense(2, activation='softmax')(z)
        return input, output



    input_layer, output_layer = get_cnn_layers()
    model = Model(input_layer, output_layer)
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=L_R), metrics=['acc'])

    #Training
    print('training model')
    global history
    history = model.fit(text_train, labels_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
            validation_data=(text_val, labels_val), verbose=2)

    return model

model = train_model(train_df, validate_df, tokenizer)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

#Generating predictions on validaton set
MODEL_NAME = 'my_model'
validate_df[MODEL_NAME] = model.predict(pad_text(validate_df['comment_text'], tokenizer))[:, 1]
validate_df.head()

#Defining AUC-based metrics and calculating bias scores on validation set predictions.

SUBGROUP_AUC = 'subgroup_auc'
BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive

def get_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan

def get_subgrp_auc(df, subgroup, label, model_name):
    subgroup_examples = df[df[subgroup]]
    return get_auc(subgroup_examples[label], subgroup_examples[model_name])


def calculate_bpsn(df, subgroup, label, model_name):
    #bpsn stands for background positive, subgroup negative, i.e we compute the score
    #of the within subgroup of negative examples and the background positive ones.

    subgroup_neg = df[df[subgroup] & ~df[label]]
    non_subgroup_pos = df[~df[subgroup] & df[label]]
    examples = subgroup_neg.append(non_subgroup_pos)
    return get_auc(examples[label], examples[model_name])

def calculate_bnsp(df, subgroup, label, model_name):
    #bsnp stands for background negative, subgroup positive. Similarly, we compute the score of
    #the within subgroup of positive examples and the background negative ones.

    subgroup_pos = df[df[subgroup] & df[label]]
    non_subgroup_neg = df[~df[subgroup] & ~df[label]]
    examples = subgroup_pos.append(non_subgroup_neg)
    return get_auc(examples[label], examples[model_name])


def compute_bias_metrics(dataset, subgroups, model, label_column, include_asegs=False):
    records = []
    for subgroup in subgroups:
        record = {
            'subgroup': subgroup,
            'subgroup_size': len(dataset[dataset[subgroup]])
        }
        record['subgroup_auc'] = get_subgrp_auc(dataset, subgroup, label_column, model)
        record['bpsn_auc'] = calculate_bpsn(dataset, subgroup, label_column, model)
        record['bnsp_auc'] = calculate_bnsp(dataset, subgroup, label_column, model)
        records.append(record)
    return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)

bias_metrics_df = compute_bias_metrics(validate_df, identities, MODEL_NAME, 'target')
print(bias_metrics_df)


# Overall scores


def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)


def overall_auc(df, model_name):
    true_labels = df['target']
    predicted_labels = df[model_name]
    return metrics.roc_auc_score(true_labels, predicted_labels)


def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_WEIGHT=0.25):
    bias_score = np.average([power_mean(bias_df['subgroup_auc'], POWER), power_mean(bias_df['bpsn_auc'], POWER),
                            power_mean(bias_df['bnsp_auc'], POWER)])
    return (OVERALL_WEIGHT * overall_auc) + ((1 - OVERALL_WEIGHT) * bias_score)


get_final_metric(bias_metrics_df, overall_auc(validate_df, MODEL_NAME))


# Predicting testing set labels

test = pd.read_csv('./data/test.csv')
submission = pd.read_csv('./data/sample_submission.csv', index_col='id')

submission['prediction'] = model.predict(pad_text(test['comment_text'], tokenizer))[:, 1]
submission.to_csv('submission.csv')