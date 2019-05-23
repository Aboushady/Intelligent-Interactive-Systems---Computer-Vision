#!/usr/bin/env python2

import numpy as np
from scipy.spatial import distance
from sklearn import preprocessing
from sklearn.externals import joblib

labels = {
    "ANGER": 0,
    "DISGUST": 1,
    "FEAR": 2,
    "HAPPY": 3,
    "SADNESS": 4,
    "SURPRISE": 5
}

def calcAllDistanceFeatures(data):
    dataPoints = len(data)
    feats = np.zeros((dataPoints,22*22))
    for i in range(dataPoints):
        index = 0
        for j in range(21):
            for k in range(21):
                if(j != k):
                    feats[i,index] = calcDist(data[j],data[k])
                    index = index+1
    return feats

def calcDist(pointA,pointB):
    return distance.euclidean(pointA,pointB)

def predConf(classifier, data):
    features = np.array([data])
    features = calcAllDistanceFeatures(data)
    features = preprocessing.normalize(features)
    features = features.reshape(features.shape[0], -1)
    prob = classifier.predict_proba(features)[0]

    index = np.where(prob == np.amax(prob))[0]
    return index, prob

def predict_emotions(landmarks):
    data = landmarks.tolist()
    classifier = joblib.load('randomForest.joblib')
    emotion, confidence = predConf(classifier, data)
    return emotion, confidence
