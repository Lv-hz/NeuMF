# %%
'''
Created on Aug 9, 2016
Keras Implementation of Multi-Layer Perceptron (GMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.  

@author: Xiangnan He (xiangnanhe@gmail.com)
'''

import numpy as np

import theano
import theano.tensor as T
import keras
from keras import backend as K
from keras import initializers
from keras.regularizers import l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Flatten, Dropout
from keras.constraints import maxnorm
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from MLP.evaluate import evaluate_model
from MLP.Dataset import Dataset
from time import time
import sys
import argparse
import multiprocessing as mp

class MLP():
    def __init__(self):
        self.load_model()
        
    def load_model(self):
        self.model = self.mlp

    def init_normal(self,shape, name=None):
        return initializers.normal(shape)

    def get_model(self,num_users, num_items, layers = [20,10], reg_layers=[0,0]):
        assert len(layers) == len(reg_layers)
        num_layer = len(layers) #Number of layers in the MLP
        # Input variables
        user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
        item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

        MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = int(layers[0]/2), name = 'user_embedding',
                                    W_regularizer = l2(reg_layers[0]),input_length=1)
        MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = int(layers[0]/2), name = 'item_embedding',
                                    W_regularizer=l2(reg_layers[0]), input_length=1)
        
        # Crucial to flatten an embedding vector!
        user_latent = Flatten()(MLP_Embedding_User(user_input))
        item_latent = Flatten()(MLP_Embedding_Item(item_input))
        
        # The 0-th layer is the concatenation of embedding layers
        vector = merge([user_latent, item_latent], mode = 'concat')
        
        # MLP layers
        for idx in range(1, num_layer):
            layer = Dense(layers[idx], W_regularizer= l2(reg_layers[idx]), activation='relu', name = 'layer%d' %idx)
            vector = layer(vector)
            
        # Final prediction layer
        prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name = 'prediction')(vector)
        
        model = Model(input=[user_input, item_input], 
                    output=prediction)
        
        return model

    def get_train_instances(self, train, num_negatives, num_items):
        user_input, item_input, labels = [],[],[]
        for (u, i) in train.keys():
            # positive instance
            user_input.append(u)
            item_input.append(i)
            labels.append(1)
            # negative instances
            for t in range(num_negatives):
                j = np.random.randint(num_items)
                while (u,j) in train:
                    j = np.random.randint(num_items)
                user_input.append(u)
                item_input.append(j)
                labels.append(0)
        return user_input, item_input, labels

    def mlp(self, guid):
        layers = eval('[64,32,16,8]')
        reg_layers = eval('[0,0,0,0]')
        num_negatives = 4
        learner = 'adam'
        learning_rate = 0.001
        batch_size = 256
        epochs = 2
        verbose = 1
        out = 1
        
        topK = 10
        evaluation_threads = 1 #mp.cpu_count()
       
        model_out_file = './PredictDir/MLP_%s.h5' %(guid)
        print('mlp {}'.format(model_out_file))
        # Loading data
        t1 = time()
        dataset = Dataset('./temp/{}/'.format(guid))
        train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
        num_users, num_items = train.shape
        print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
            %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))
        
        # Build model
        model = self.get_model(num_users, num_items, layers, reg_layers)
        if learner.lower() == "adagrad": 
            model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
        elif learner.lower() == "rmsprop":
            model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
        elif learner.lower() == "adam":
            model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
        else:
            model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')    
        
        # Check Init performance
        t1 = time()
        (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        print('Init: HR = %.4f, NDCG = %.4f [%.1f]' %(hr, ndcg, time()-t1))
        
        # Train model
        best_hr, best_ndcg, best_iter = hr, ndcg, -1
        for epoch in range(epochs):
            t1 = time()
            # Generate training instances
            user_input, item_input, labels = self.get_train_instances(train, num_negatives, num_items)
        
            # Training        
            hist = model.fit([np.array(user_input), np.array(item_input)], #input
                            np.array(labels), # labels 
                            batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
            t2 = time()

            # Evaluation
            if epoch %verbose == 0:
                (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
                hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
                print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                    % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
                if hr > best_hr:
                    best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                    if out > 0:
                        model.save(model_out_file, overwrite=True)

        print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
        if out > 0:
            print("The best MLP model is saved to %s" %(model_out_file))
        return model_out_file