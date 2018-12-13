import MFCC
import librosa
import numpy as np
import os
import sklearn.mixture
import sys
import glob

def fit(frames, test_ratio=0.5, n_components=11):
    index = np.arange(len(frames))
    np.random.seed(0)
    np.random.shuffle(index)

    train_idx = index[int(len(index) * test_ratio):]
    test_idx = index[:int(len(index) * test_ratio)]

    gmm = sklearn.mixture.GaussianMixture(n_components=n_components)
    gmm.fit(frames[train_idx])

    return gmm, frames[test_idx]

def predict(gmms, test_frame):
    scores = []
    for gmm_name, gmm in gmms.items():
        scores.append((gmm_name, gmm.score(test_frame)))
    return sorted(scores, key=lambda x: x[1], reverse=True)

def evaluate(gmms, test_frames):
    correct = 0

    for name in test_frames:
        best_name, best_score = predict(gmms, test_frames[name])[0]
        print 'Ground Truth: %s, Predicted: %s' % (name, best_name)
        if name == best_name:
            correct += 1

    print 'Overall Accuracy: %f' % (float(correct) / len(test_frames))

if __name__ == '__main__':
    gmms, test_frames = {}, {}
    for t in range(11):
        name = t
        print 'Processing %s ...' % name

        i = 0
        for filename in glob.glob('wavFiles/'+str(t)+'/'+str(i)+'.wav'):
            if i == 0:
                mfccs = MFCC.mfcc(filename,13)
            else:
                mfccs = np.hstack((mfccs, MFCC.mfcc(filename,13)))
            i += 1
            # name = os.path.splitext(os.path.basename(filename))[0]
        gmms[name], test_frames[name] = fit(mfccs.T)

    evaluate(gmms, test_frames)
