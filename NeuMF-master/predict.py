import h5py
import csv
import os
from MLP.PredictDataset import PredictDataset
from keras.models import load_model
import math
import heapq # for retrieval topK
import multiprocessing
import numpy as np
from time import time


class Predict():
    def __init__(self):
        self.load_model()
        
    def load_model(self):
        self.model = self.predict

    def predictItem(self,model, testRatings, testNegatives, K, guid): 
        for idx in range(len(testRatings)):
            self.predictDetail(guid,K,idx)    
        return 0

    def predictDetail(self,guid,K,idx):
        rating = testRatings[idx]
        items = testNegatives[idx]
        u = rating[0]
        gtItem = rating[1]
        items.append(gtItem)
        map_item_score = {}
        users = np.full(len(items), u, dtype = 'int32')
        predictions = model.predict([users, np.array(items)], 
                                    batch_size=100, verbose=0)
        for i in range(len(items)):
            item = items[i]
            map_item_score[item] = predictions[i]
        items.pop()
        
        # heapq.nlargest:从迭代器对象iterable中返回前n个最大的元素列表
        ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get) 
        if not os.path.isdir('./PredictDir/{}'.format(guid)):
            os.makedirs('./PredictDir/{}'.format(guid))
        with open('./PredictDir/{}/{}.csv'.format(guid,guid), mode='a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            newRanklist = [u] + ranklist
            print(newRanklist)
            writer.writerow(newRanklist)

        return 0

    def predict(self, guid):
        global model
        model = load_model('./PredictDir/MLP_%s.h5' %(guid))
        K=10
        predictDataset = PredictDataset('./PredictDir/{}/'.format(guid))
        global testRatings, testNegatives
        testRatings, testNegatives = predictDataset.testRatings, predictDataset.testNegatives
        self.predictItem(model,testRatings, testNegatives, K, guid)
        predictPath = './PredictDir/{}/{}.csv'.format(guid,guid)
        print(predictPath)
        return predictPath