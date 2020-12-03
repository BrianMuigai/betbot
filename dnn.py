import keras.callbacks as cb
from keras.datasets import mnist
from keras.layers.core import Activation, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD,Adam
from keras.regularizers import l1, l2
from keras.utils import np_utils
from keras.models import model_from_json
from matplotlib import pyplot as plt
import numpy as np
import time
import os
import pandas as pd

loc = os.getcwd() +'/data/'

def PreprocessDataset():
    from sklearn import preprocessing
    X_features = ['HST','AST','HAS','HDS','AAS','ADS','HC','AC','HTWS','HTDS','HTLS','ATWS','ATDS','ATLS',
    'HW','HD','HL','AW','AD','AL','HGS','AGS','HGC','AGC','WD','DD','LD','HF','AF','MR','MW','CornerDiff',
    'GoalsScoredDiff','GoalsConceedDiff','ShotsDiff','HomeTeamLP','AwayTeamLP','PD','RD','DAS','DDS'
    ,'B365H','B365D','B365A','probHome','probDraw','probAway',]

    data = pd.read_csv(loc + 'england/training_dataset.csv')
    data = data.reindex(np.random.permutation(data.index))

    def transformResult(row):
        '''Converts results (H,A or D) into numeric values'''
        if(row.FTR == 'H'):
            return 1
        elif(row.FTR == 'A'):
            return 2
        else:
            return 0

    data["result"] = data.apply(lambda row: transformResult(row),axis=1)
    
    x = data[X_features]
    y = data['result']
    
    train_max_row = int(data.shape[0]*0.9)
    
    x_train = x.iloc[:train_max_row]
    x_test = x.iloc[train_max_row:]
    
    y_train = y.iloc[:train_max_row]
    y_test = y.iloc[train_max_row:]
    
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    
    ################Pre-processing###########
    x_train = preprocessing.scale(x_train)
    x_test = preprocessing.scale(x_test)
    
    return x_train, x_test, y_train, y_test

def DefineModel():
    activation_func = 'relu' 
    loss_function = 'categorical_crossentropy'
    #loss_function = 'mean_squared_error'
      
    dropout_rate = 0.4
    weight_regularizer = None
    learning_rate = 0.005
    
    ## Initialize model.
    model = Sequential()
    ## 1st Layer
    ## Dense' means fully-connected.
    model.add(Dense(128, input_dim=47, W_regularizer=weight_regularizer))
    model.add(Activation(activation_func))
    model.add(Dropout(0.5))
    
    ## 2nd Layer
    model.add(Dense(64, input_dim=128, W_regularizer=weight_regularizer))
    model.add(Activation(activation_func))
    model.add(Dropout(dropout_rate))
    
    ## 3rd Layer
    model.add(Dense(32))
    model.add(Activation(activation_func))   
    model.add(Dropout(dropout_rate))
    
    ## 4th Layer
    model.add(Dense(16))
    model.add(Activation(activation_func))   
    model.add(Dropout(dropout_rate))
    
    ## 5th Layer
    model.add(Dense(8))
    model.add(Activation(activation_func))   
    model.add(Dropout(dropout_rate))
    
    ## Adding Softmax Layer
    ## Last layer has the same dimension as the number of classes
    model.add(Dense(3))
    
    ## For classification, the activation is softmax
    model.add(Activation('softmax'))
    
    ## Define optimizer. we select Adam
    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #opt = SGD(lr=learning_rate, clipnorm=5.)
    
    ## Define loss function = 'categorical_crossentropy' or 'mean_squared_error'
    model.compile(loss=loss_function, optimizer=opt, metrics=["accuracy"])
    return model


x_train, x_test, y_train, y_test = PreprocessDataset()

def TrainModel(data=None, epochs=20):
    batch=100
    start_time = time.time()
    exists = os.path.isfile(loc+'model.json') and os.path.isfile(loc+'model.h5')
    model = None
    if exists:
        model = load_model()
    else:
        model = DefineModel()
    if data is None:
        print("Must provide data.")
        return
    x_train, x_test, y_train, y_test = data
    print('Start training.')
    _offset = int(len(x_train)*0.85)
    ## Use the first 90% samples to train, last 10* samples to validate.
    history = model.fit(x_train[:_offset], y_train[:_offset], nb_epoch=epochs, batch_size=batch,
              validation_data=(x_train[_offset:], y_train[_offset:]))
    print("Training took {0} seconds.".format(time.time() - start_time))
    return model, history

trained_model, training_history = TrainModel(data=[x_train, x_test, y_train, y_test],epochs = 50)

def TestModel(model=None, data=None):
    if model is None:
        print("Must provide a trained model.")
        return
    if data is None:
        print("Must provide data.")
        return
    x_test, y_test = data
    scores = model.evaluate(x_test, y_test)
    return scores

test_score = TestModel(model=trained_model, data=[x_test, y_test])
print("Test loss {:.4f}, accuracy {:.2f}%".format(test_score[0], test_score[1] * 100))

def saveModel(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

def load_model():
    json_file = open(loc+'model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(loc+"model.h5")
    print("Loaded model from disk")
    return loaded_model

