import librosa  
import numpy as np
import kapre
from keras.models import Sequential, load_model
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D
from kapre.augmentation import AdditiveNoise
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation
from keras.callbacks import ModelCheckpoint
import os

sampleCount=44100
channelCount=1
trainDataPath="data/train/"
testDataPath="data/test/"
modelName='bugModel.h5'

def get_class_names(path=trainDataPath):
        class_names = os.listdir(path)
        return class_names


def getTrainCount():
        i=0
        for className in  os.listdir(trainDataPath) :
                for fileName in os.listdir(trainDataPath+className+"/"):
                        i=i+1
        return i


def encode_class(class_name, class_names):  
        try:
                idx = class_names.index(class_name)
                vec = np.zeros(len(class_names))
                vec[idx] = 1
                return vec
        except ValueError:
                return None
        

def  loadData(path) :
        trainIndex=0
        allClass=get_class_names()
        trainCount=getTrainCount()
        retX=np.zeros((trainCount,channelCount,sampleCount))
        retY=np.zeros((trainCount,len(allClass)))
        for className in  os.listdir(path) :
                for fileName in os.listdir(path+className+"/"):
                        fullFileName="{}/{}/{}".format(path,className,fileName)
                        samples,sr= librosa.load (fullFileName,sr=44100)
                        samples=samples[np.newaxis,np.newaxis,0:sampleCount]
                        npad=((0,0),(0,0),(0,sampleCount-samples.shape[2]))
                        retX[trainIndex,:,:]=np.pad(samples,pad_width=npad,mode='constant', constant_values=0)
                        y_val=encode_class(className, allClass)
                        retY[trainIndex,:]=y_val
        return retX,retY

def train():
        pool_size = (2, 2) 
        # 350 samples
        input_shape = (channelCount, sampleCount) 
        sr = 44100
        model = Sequential()
        model.add(Melspectrogram(n_dft=512, n_hop=256, input_shape=input_shape,
                                 padding='same', sr=sr, n_mels=128,
                                 fmin=0.0, fmax=sr/2, power_melgram=1.0,
                                 return_decibel_melgram=False, trainable_fb=False,
                                 trainable_kernel=False,
                                 name='trainable_stft'))
        model.add(AdditiveNoise(power=0.2))
        model.add(Normalization2D(str_axis='freq')) # or 'channel', 'time', 'batch', 'data_sample'
        
        
        model.add(Convolution2D(32, 3, 3))
        model.add(BatchNormalization(axis=1 ))
        model.add(ELU(alpha=1.0))  
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))        
        
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(get_class_names())))
        model.add(Activation("softmax"))        
        
        model.compile('adam', 'categorical_crossentropy') 
        
        x,y=loadData(trainDataPath)
        
        
        
        checkpoint_filepath = 'weights.hdf5'
        print("Looking for previous weights...")
        if ( os.path.isfile(checkpoint_filepath) ):
                print ('Checkpoint file detected. Loading weights.')
                model.load_weights(checkpoint_filepath)
        else:
                print ('No checkpoint file detected.  Starting from scratch.')

        checkpointer = ModelCheckpoint(filepath=checkpoint_filepath, verbose=1, save_best_only=True)        
        test_x,test_y=loadData(testDataPath)
        
        model.fit(x, y,batch_size=128, nb_epoch=100,verbose=1,validation_data=(test_x, test_y), callbacks=[checkpointer])
        
        model.save(modelName)


def test():
        model = load_model(modelName, custom_objects={'Melspectrogram':kapre.time_frequency.Melspectrogram,'AdditiveNoise':kapre.augmentation.AdditiveNoise,'Normalization2D':kapre.utils.Normalization2D})        
        x,y=loadData(testDataPath)
        score=model.evaluate(x,y, verbose=0)
        print('Test score:', score)
        #print('Test accuracy:', score[1])        

if __name__ == '__main__':
        train()
        test()