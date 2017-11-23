# keras_audio_classifier
首先准备好python环境，下个anaconda,大部分需要的python库就具备了，接着安装tensorflow和keras.一路pip，缺什么就装什么，记得使用anaconda的pip。

关于声音，一般我们会说，这段音频是：2声道，16位采样，采样率为44.1k,这代表啥意思呢，这其实表达了声音的三要素：
1、采样率：自然界的声音环境其实是连续的，全部采集的话会很大而且也没有必要，所以要采用抽样的方式来采集声音，采样率就是以秒为单位的抽样次数，比如44.1k,就代表一秒内采了44.1k次。
2、采样宽度：采集到的点的范围可以是精度很高，也可以比较低，其实就是数值，比如double占的空间比int要大，主要是double的能表达范围精度比int高。同理，每个采样点采集回来以后要换算成一个数值存储起来，这里也有精度考虑，一般16位来存储就差不多了，也可以高或低，按个人喜好。
3、声道：也就是同时有几个声道录制，比如单声道也叫mono,双声道。。。

声音的矩阵表示：
1、采样率为44.1k，声道为1声道的一秒的声音转成的二维数组。
[[1,2,3.....44100]]

2、采样率为44.1k，声道为2声道的一秒的声音转成的二维数组。
[[1,2,3.....44100]
[1,2,3.....44100]]

声道一个维度，采样点一个维度，已经有两个维度了。如果交给机器学习，我们的训练数据还要再加一个维度，就是训练次数，于是我们的准备数据是一个三维数组，
1、代表有100个样本，一个声道，采样率是44.1k的一秒数据。
（100，1，44100）
2、代表有10个样本，5个声道，采样率是44.1k的两秒数据。
（10，5，88200）

以上的说明表达了准备数据的格式。要根据具体的情况进行数据整理组织。

利用librosa进行数据提取：
librosa是一个python音频处理的一个包，它其实只是一个wrapper,是需要后端，后端应该是ffmpeg.简单的使用它：
      samples,sr= librosa.load (fullFileName,mono=False,sr=44100)
这样取到的samples就是我们想要的二维数组。分别是通道与采样点两维。我们可以通过加维将二维变成三维，使之可以满足我们的训练要求：
samples=samples[np.newaxis,:,:]
另外，如果存在采样数据不足的情况下可以使用np.pad进行填0.总之让这个数据统一。


利用kapre进行训练：
kapre是github上的一个音频预处理包，可以通过pip进行安装。

它的简单模型如下：
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
        model.add(Normalization2D(str_axis='freq')) 
        
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

采用的是梅尔声谱作为主要层，还AdditiveNoise增加噪声，最后再拉平输出分类结果。中间可以根据需要添加卷积层（自己随便加）如：
        model.add(Convolution2D(32, 3, 3))
        model.add(BatchNormalization(axis=1 ))
        model.add(ELU(alpha=1.0))  
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))    

最后来使用模型：
def test():
        model = load_model(modelName, custom_objects={'Melspectrogram':kapre.time_frequency.Melspectrogram,'AdditiveNoise':kapre.augmentation.AdditiveNoise,'Normalization2D':kapre.utils.Normalization2D})        
        x,y=loadData(testDataPath)
        score=model.evaluate(x,y, verbose=0)
        print('Test score:', score)
