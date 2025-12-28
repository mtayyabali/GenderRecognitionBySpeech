import os
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import wave
import numpy as np
from python_speech_features import mfcc
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.external import joblib
from sklearn.neural_network import MLPClassifier
import random

mlp = MLPClassifier(solver='adam', alpha=1e-5,
                  hidden_layer_sizes=(50), random_state=2, max_iter=100000, warm_start=True)
def get_MFCC(sr, audio):
    features = preprocessing.scale(mfcc(audio, sr, 0.025, 0.01, 100, appendEnergy=False))
    return features

def getRondomList(len_feature):
    i = 0
    random_list = []
    while i < 100:
        temp = random.randint(0, len_feature)
        if temp in random_list:
            continue
            # print('...')
        else:
            random_list.append(temp)
            i = i + 1
    return random_list


def train_mlp(data_path, dtype):
    male_train_path = os.path.join(data_path, 'male_train_set_256')
    female_train_path = os.path.join(data_path, 'female_train_set_256')
    if not os.path.isdir(male_train_path) or not os.path.isdir(female_train_path):
        raise Exception("illegal directory path.")
    file_list_male = [os.path.join(male_train_path, f) for f in os.listdir(male_train_path) if f.endswith('.wav')]
    file_list_female = [os.path.join(female_train_path, f) for f in os.listdir(female_train_path) if f.endswith('.wav')]

    mfcc_features = []
    for file in file_list_male:
        f = wave.open(file, 'rb')
        print(file)
        frame_rate, n_frames = f.getframerate(), f.getnframes()
        if n_frames == 0:
            print(file)
            continue
        audio = np.fromstring(f.readframes(n_frames), dtype=dtype)
        feature = get_MFCC(frame_rate, audio)
        random_list = getRondomList(len(feature)-1)
        for i in random_list:
            mfcc_features.append(feature[i])
    n_frames_male = len(mfcc_features)
    print(n_frames_male)

    for file in file_list_female:
        f = wave.open(file, 'rb')
        print(file)
        frame_rate, n_frames = f.getframerate(), f.getnframes()
        if n_frames == 0:
            print(file)
            continue
        audio = np.fromstring(f.readframes(n_frames), dtype=dtype)
        feature = get_MFCC(frame_rate, audio)
        random_list = getRondomList(len(feature) - 1)
        for i in random_list:
            mfcc_features.append(feature[i])

    n_frames_female = len(mfcc_features) - n_frames_male
    print(n_frames_female)
    print("female")
    Y1 = [0.0]                                    # male
    Y2 = [1.0]                                   # female
    Y1 = Y1 * n_frames_male
    Y2 = Y2 * n_frames_female
    y = Y1 + Y2
    mlp.fit(mfcc_features, y)
    print('now saving..')
    joblib.dump(mlp, 'test_round_2.pkl')


def test_mlp(data_path, dtype):
    male_test_path = os.path.join(data_path, '256_male_test')
    female_test_path = os.path.join(data_path, '256_female_test')
    if not os.path.isdir(male_test_path) or not os.path.isdir(female_test_path):
        raise Exception("illegal directory path.")
    file_list_male = [os.path.join(male_test_path, f) for f in os.listdir(male_test_path) if f.endswith('.wav')]
    file_list_female = [os.path.join(female_test_path, f) for f in os.listdir(female_test_path) if f.endswith('.wav')]

    mlp = joblib.load('test_round_2.pkl')
    false_positive_female = 0
    false_positive_male = 0
    i = 0
    accuracy_male = 0
    for file in file_list_male:
        f = wave.open(file, 'rb')
        frame_rate, n_frames = f.getframerate(), f.getnframes()
        if n_frames == 0:
            i = i + 1
            print(file)
            continue
        audio = np.fromstring(f.readframes(n_frames), dtype=dtype)
        feature = get_MFCC(frame_rate, audio)
        #print(feature)
        test_feature = []
        random_list = getRondomList(len(feature) - 1)
        for k in random_list:
            test_feature.append(feature[k])
        scores = mlp.predict(test_feature)
        #print(scores)
        print(file)
        print(np.sum(scores == 0), '--', np.sum(scores == 1))
        if (np.sum(scores == 0)) >= (np.sum(scores == 1)):
            accuracy_male = accuracy_male + 1
        if (np.sum(scores == 0)) <= (np.sum(scores == 1)):
            false_positive_female = false_positive_female + 1

    j = 0
    accuracy_female = 0
    for file in file_list_female:
        f = wave.open(file, 'rb')
        frame_rate, n_frames = f.getframerate(), f.getnframes()
        if n_frames == 0:
            j = j + 1
            print(file)
            continue
        audio = np.fromstring(f.readframes(n_frames), dtype=dtype)
        feature = get_MFCC(frame_rate, audio)
        #print(feature)
        test_feature = []
        random_list = getRondomList(len(feature) - 1)
        for k in random_list:
            test_feature.append(feature[k])
        scores = mlp.predict(test_feature)
        print(file)
        #print(scores)
        print(np.sum(scores == 0), '--', np.sum(scores == 1))
        if (np.sum(scores == 0)) <= (np.sum(scores == 1)):
            accuracy_female = accuracy_female + 1
            if (np.sum(scores == 0)) >= (np.sum(scores == 1)):
                false_positive_male = false_positive_male + 1
    print("precision:")
    print("female: ", (accuracy_female / ((len(file_list_female) - j) + false_positive_female)) * 100)
    print("male: ", (accuracy_male / ((len(file_list_female) - j) + false_positive_male)) * 100)
    print("accuracy: ")
    print("female:", (accuracy_female / (len(file_list_female) - j)) * 100)
    print("male:", (accuracy_male / (len(file_list_male) - i)) * 100)
    print("Recall:")
    print("female: ", accuracy_female / ((len(file_list_female) - j) + ((len(file_list_female) - j) - accuracy_female)) * 100)
    print("male: ", accuracy_male / ((len(file_list_male) - j) + ((len(file_list_male) - j) - accuracy_male)) * 100)


if __name__ == '__main__':
    train_mlp("./pre_data", np.int16)
    test_mlp('./data', np.int16)
