import numpy as np
import librosa
import os
import pickle
import sys


def readMfcc(audioPath, size, length):
    featuresArray = []
    for i in range(0, size, length):
        if i + length <= size - 1:
            y, sr = librosa.load(audioPath, offset=i / length, duration=0.1)  #Instead of"length" used to be 0.1f
            S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
            log_S = librosa.logamplitude(S, ref_power=np.max)
            mfcc = librosa.feature.mfcc(S=log_S, sr=sr, n_mfcc=13)
            featuresArray.append(mfcc)
    return np.reshape(featuresArray,(len(featuresArray),13*5),order='C')

def getLabel(fileName, label_dict):
    genre = label_dict[fileName.split('.')[0]]
    ans = np.zeros(len(label_dict))
    ans[genre] = 1;
    return ans

def saveDataIn(path, data):
    with open(path, 'wb') as f:
        f.write(pickle.dumps(data))

def getDataFrom(path):
    data = []
    with open(path, 'rb') as f:
        content = f.read()
        data = pickle.loads(content)
    return data


if (len(sys.argv) < 3):
    raise Exception("You must provide the path to the data and the output path\npython " + sys.argv[0] + " path_to_data output_path")

path = sys.argv[1]
sample_length = 1000

label_dict ={
    "blues": 0,
    "classical": 1,
    "country": 2,
    "disco": 3,
    "hiphop": 4,
    "jazz": 5,
    "metal": 6,
    "pop": 7,
    "reggae": 8,
    "rock": 9,

}


inputs = []
labels = []
c = 0
for root, subdirs, files in os.walk(path):
    for filename in files:
        if filename.endswith("au"):
            c += 1
            file_path = os.path.join(root, filename)
            #audio = AU(file_path)
            #print("{}:{}".format(int(30/60),(int(30%60))))
            print("{}%".format(c/len(files)))
            inputs.append(readMfcc(file_path,int(30*1000), sample_length))
            for i in range(29):
                labels.append(getLabel(filename, label_dict))

output_path = sys.argv[2]
saveDataIn(output_path + "/inputs", inputs)
saveDataIn(output_path + "/labels", labels)
