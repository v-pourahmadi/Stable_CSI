# -*- coding: utf-8 -*-


import copy
import keras

from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Reshape, Lambda, concatenate, BatchNormalization, Conv2D, MaxPooling2D, Dropout, Conv1D, MaxPooling1D, Add, Subtract
from keras import regularizers
from keras.initializers import RandomNormal
from keras.callbacks import LambdaCallback
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU, PReLU

from keras.optimizers import Adam, Nadam
import numpy as np
from keras import backend as K
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.externals import joblib
import tensorflow as tf
import os
#import matplotlib
from tensorflow.python.lib.io import file_io
# import cloudstorage as gcs


# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# matplotlib.use('Agg')

activation_fun='relu'
activation_fun_2='LeakyReLU'

#activation_fun='tanh'

droprate=0.01



def save_file_toCLoud(filename, Save_loc, save_name):
    print(filename)
    print(Save_loc)
    print(save_name)
    with file_io.FileIO(filename, mode='rb') as input_f:
        with file_io.FileIO(Save_loc+save_name, mode='wb+') as output_f:
            output_f.write(input_f.read())
            print("model/file Saved to GCS")

def Copy_file_fromCLoud(loadfilename, save_name):
    with file_io.FileIO(loadfilename, mode='rb') as input_f:
        with file_io.FileIO(save_name, mode='wb+') as output_f:
            output_f.write(input_f.read())
            print("model Saved localy")

def SaveModelToCloud(model,Save_loc,save_name):
    print("5")
    model.save(save_name)
    with file_io.FileIO(save_name, mode='rb') as input_f:
        with file_io.FileIO(Save_loc+"/"+save_name, mode='wb+') as output_f:
            output_f.write(input_f.read())

def savemodeltocloud_local(epoch,steps, self,model,savename):
    if (epoch % steps) == 0:
        print("4")
        SaveModelToCloud(model,self.log_path, savename+'{:03d}'.format(epoch)+'.hdf5')


class Det_satable():

    # def get_my_MSE_loss():
    #     def my_MSE(y_true, y_pred):
    #         return K.mean(K.square(y_pred - y_true))
    #     return my_MSE



    def __init__(self, encoded_dim=10, on_cloud=1, test_mode=0, just_trained=0, log_path='.', normalize_mode=2, 
                 data_type=0, train_prev_model=".", batch_size=36, input_shape=(60,100,1),if_real_imag=2):
        self.encoded_dim=encoded_dim
        #self.optimizer = Adam(lr=10**-2,decay=.85)
        self.optimizer = Nadam()
        self.test_mode=test_mode        
        self.just_trained=just_trained 
        self.batch_size=batch_size
        self.log_path=log_path        
        self.on_cloud=on_cloud
        self.normalize_mode=normalize_mode
        self.data_type=data_type
        self.train_prev_model=train_prev_model
        self.input_shape=input_shape
        self.if_real_imag=if_real_imag
        print('======')
        print(self.input_shape)
        self._initAndCompileFullModel(encoded_dim)
        #self.scaler = StandardScaler(with_mean=True, with_std=True)
 
    def _genEncoder2D(self,In_shape):
        act = LeakyReLU(alpha=0.3)
        #act = LeakyReLU(alpha=1e-3)
        encoder = Sequential()
        if self.data_type==3 or  self.data_type==4:
            encoder.add(Conv1D(filters=32,
                 kernel_size=3,
                 #activation=activation_fun,
                 #kernel_regularizer=regularizers.l2(5e-6),
                 input_shape=In_shape,
                 padding='same',
                 strides=1))
            encoder.add(act)
            encoder.add(BatchNormalization())
            #encoder.add(MaxPooling1D(pool_size=2))
            encoder.add(Dropout(droprate))

            encoder.add(Conv1D(filters=64,
                 kernel_size=3,
                 #activation=activation_fun,
                 #kernel_regularizer=regularizers.l2(5e-6),
                 #input_shape=self.input_shape,
                 padding='same',
                 strides=1))
            encoder.add(act)            
            encoder.add(BatchNormalization())
            #encoder.add(MaxPooling1D(pool_size=2))
            encoder.add(Dropout(droprate))

            encoder.add(Conv1D(filters=4,
                 kernel_size=3,
                 #activation=activation_fun,
                 #kernel_regularizer=regularizers.l2(5e-6),
                 #input_shape=self.input_shape,
                 padding='same',
                 strides=1))
            encoder.add(act)            
            encoder.add(BatchNormalization())
            #encoder.add(MaxPooling1D(pool_size=2))
            #encoder.add(Dropout(droprate))

        encoder.summary()
        return encoder

    def _genCompresor(self,In_shape):
        Compresor = Sequential()

        Compresor.add(Flatten(name='conv1',input_shape=In_shape))

        Compresor.add(Dense(100, activation=activation_fun))
        Compresor.add(BatchNormalization())
        Compresor.add(Dropout(droprate))

        Compresor.add(Dense(50, activation=activation_fun))
        Compresor.add(BatchNormalization())
        Compresor.add(Dropout(droprate))

        Compresor.add(Dense(25, activation=activation_fun))
        Compresor.add(BatchNormalization())
        Compresor.add(Dropout(droprate))

        Compresor.add(Dense(self.encoded_dim, activation=activation_fun))
        #encoder.add(BatchNormalization())

        Compresor.summary()
        return Compresor


    def _genClassifier(self):
        Classifier = Sequential()
        # #single_dense.add(Dense(np.prod(img_shape),input_shape=input_dim))
        # single_dense.add(Reshape(img_shape,input_shape=input_dim))

        Classifier.add(Dense(self.encoded_dim, activation=activation_fun,input_shape=(2*self.encoded_dim,)))
        #Classifier.add(BatchNormalization())
        Classifier.add(Dense(6, activation=activation_fun))
        Classifier.add(Dense(4, activation=activation_fun))
        #Classifier.add(BatchNormalization())
        #Classifier.add(Dense(2, activation='softmax'))
        #Classifier.add(Dense(1, activation='softmax'))
        Classifier.add(Dense(1, activation='sigmoid'))

        Classifier.summary()
        return Classifier

    def _initAndCompileFullModel(self, encoded_dim):

        Conv_encoder=0
        if Conv_encoder==1:
            self.Encoder_p1=self._genEncoderConv()
        else:
            if self.if_real_imag==0 or self.if_real_imag==1:
                self.Encoder_p1=self._genEncoder()
            if self.if_real_imag==2:
                self.Encoder_p1_1=self._genEncoder2D(In_shape=self.input_shape)
                self.Encoder_p1_2=self._genEncoder2D(In_shape=(self.input_shape[0],4))
                self.Compresor_p1=self._genCompresor(In_shape=(self.input_shape[0],4))
                #self.Encoder_p2=self._genEncoder2D()

        #self.Encoder_p2=self._genEncoder()

        self.Classifier_p=self._genClassifier()

        if self.data_type==1:
            input_dim=60
        elif self.data_type==2:
            input_dim=120
        elif self.data_type==3 or  self.data_type==4:
            input_dim=self.input_shape

        print(input_dim)

        CSI_Data_of_ref = Input(shape=input_dim)
        CSI_Data_of_check = Input(shape=input_dim)

        # if self.normalize_mode==2:
        #     noisy_image_n = Lambda(self.my_normalize)(noisy_image)
        # else:
        #     noisy_image_n=noisy_image
        # selected_img= self.selector(noisy_image_n)

        encode_ref_1= self.Encoder_p1_1(CSI_Data_of_ref)
        #encode_ref_2= And () ([CSI_Data_of_ref,encode_ref_1])
        encode_ref_2= self.Encoder_p1_2(encode_ref_1)
        #encode_ref_2= encode_ref_1
        encode_ref= self.Compresor_p1(encode_ref_2)

        encode_check_3= self.Encoder_p1_1(CSI_Data_of_check)
        #encode_check_2= Add () ([CSI_Data_of_check,encode_check_1])
        encode_check_4= self.Encoder_p1_2(encode_check_3)
        #encode_check_4= encode_check_3
        encode_check= self.Compresor_p1(encode_check_4)

        encode_csi = concatenate([encode_ref, encode_check])

        Class_des = self.Classifier_p(encode_csi)

        self.Dens_estim = Model([CSI_Data_of_ref,CSI_Data_of_check], Class_des)
        #self.Dens_estim.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        #self.Dens_estim.compile(optimizer=self.optimizer, loss='mse', metrics=['accuracy'])
        self.Dens_estim.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        ##self.Dens_estim.compile(optimizer=self.optimizer, loss='categorical_hinge', metrics=['accuracy'])

        if self.test_mode==1:
            # error()
            if self.on_cloud==0:
                # if self.Enable_conv==1:
                #     Weigth_data_2=self.log_path+"/"+"weights_2.hdf5"
                #     if (os.path.isfile(Weigth_data_2)):
                #         print("Conv loaded")
                #         self.autoencoder_convonly.load_weights(Weigth_data_2) 
                if self.just_trained==1:
                    Weigth_data="weights.hdf5"
                    if (os.path.isfile(Weigth_data)):
                        self.Dens_estim.load_weights(Weigth_data)
                        print("loaded weights")
                    else:
                        print("train the model first!!!")

                else:
                    Weigth_data=self.train_prev_model+"/"+"weights.hdf5"
                    print(Weigth_data)
                    if (os.path.isfile(Weigth_data)):
                        self.Dens_estim.load_weights(Weigth_data)
                        print("loaded weights")
                    else:
                        print("train the model first!!!")
                    # Weigth_data_1=self.train_prev_model+"/"+"weights_1.hdf5"
                    # if (os.path.isfile(Weigth_data_1)):
                    #     print("Dense loaded")
                    #     self.Dens_estim.load_weights(Weigth_data_1) 

            else:
                print("Ha1")
                if self.just_trained==1:
                    Weigth_data="weights.hdf5"
                    if (os.path.isfile(Weigth_data)):
                        self.Dens_estim.load_weights(Weigth_data)
                        print("loaded weights")
                    else:
                        print("train the model first!!!")
                else:
                    #might be changed if the weights location changes
                    Weigth_data=self.train_prev_model+"/"+"weights.hdf5"
                    print(Weigth_data)
                    if file_io.file_exists(Weigth_data):
                        Copy_file_fromCLoud(Weigth_data,"weights.hdf5")
                        self.Dens_estim.load_weights("weights.hdf5")
                        print("all loaded")
                    else:
                        print("train the model first_1!!!")


    def train(self, x_in1, x_in2, y_in, input1_t, input2_t,target2_t, batch_size=32, epochs=5):


        if self.data_type==1 or  self.data_type==2 or  self.data_type==3 or  self.data_type==4:
            x1_scaled=x_in1
            x2_scaled=x_in2

            # print(y_in.shape)
            # print(y_in[1:10,:])
            # # print(y_in[1:10,0])
            # y_in_on=np_utils.to_categorical(1-y_in[:,0], num_classes=2)
            # # print(y_in_on.shape)
            # print(y_in_on[1:10,:])
            # print(y_in_on[1:10,0])
            #y_onehot=y_in_on
            y_onehot=y_in
            #y_onehot=y_in[:,0]


            # print(target2_t.shape)
            # print(target2_t[1:10,:])
            # print(target2_t[1:10,0])
            # target2_t_o=np_utils.to_categorical(1-target2_t[:,0], num_classes=2)
            # print(target2_t_o.shape)
            # print(target2_t_o[1:10,:])
            # print(target2_t_o[1:10,0])
            #target2_t_onehot=target2_t_o
            target2_t_onehot=target2_t
            #target2_t_onehot=target2_t[:,0]

            Weigth_data=self.train_prev_model+"/"+"weights.hdf5"
            print("tries to load model.")
            if file_io.file_exists(Weigth_data):
                print(Weigth_data)
                if self.on_cloud==1:
                    Copy_file_fromCLoud(Weigth_data,"weights.hdf5")
                    self.Dens_estim.load_weights("weights.hdf5")
                    print("all loaded")
                else:
                    self.Dens_estim.load_weights(Weigth_data)
                    print("all loaded")

            STEPs=3
            if self.on_cloud==1:
                patience_val=50
            else:
                patience_val=50

            earlyStopping_1=keras.callbacks.EarlyStopping(monitor='val_acc', patience=patience_val, verbose=0, mode='auto')

            if self.on_cloud==1:
                epoch_save_model_1=LambdaCallback(on_epoch_end=lambda epoch, logs: savemodeltocloud_local(epoch,STEPs,self, self.Dens_estim,'weights'))
                # self.Dens_estim.fit([x1_scaled,x2_scaled], y_scaled, epochs=epochs, batch_size=batch_size, shuffle=True,validation_split=0.3,
                #                       callbacks=[earlyStopping_1,
                #                                 keras.callbacks.ModelCheckpoint('weights.hdf5', 
                #                                    verbose=0, 
                #                                    monitor='val_loss',
                #                                    #save_best_only=False, 
                #                                    save_best_only=True, 
                #                                    save_weights_only=False, 
                #                                    mode='auto', 
                #                                    period=1),
                #                                 epoch_save_model_1
                #                                 ])
                self.Dens_estim.fit([x1_scaled,x2_scaled], y_onehot, epochs=epochs, batch_size=self.batch_size, shuffle=True,
                                        #validation_data=([input1_t,input2_t], target2_t_onehot),
                                        validation_split=0.05,                                      
                                      callbacks=[earlyStopping_1,
                                                keras.callbacks.ModelCheckpoint('weights.hdf5', 
                                                   verbose=0, 
                                                   monitor='val_acc',
                                                   #save_best_only=False, 
                                                   save_best_only=True, 
                                                   save_weights_only=False, 
                                                   mode='auto', 
                                                   period=1),
                                                epoch_save_model_1
                                                ])
                save_file_toCLoud('weights.hdf5', self.log_path, '/weights.hdf5')
                SaveModelToCloud(self.Dens_estim,self.log_path, '/weights_.hdf5')

            else:
                # self.Dens_estim.fit([x1_scaled,x2_scaled], y_scaled, epochs=epochs, batch_size=batch_size, shuffle=True,validation_split=0.3,
                #                       callbacks=[earlyStopping_1,
                #                                 keras.callbacks.ModelCheckpoint('weights.hdf5', 
                #                                    verbose=0, 
                #                                    monitor='val_loss',
                #                                    #save_best_only=False, 
                #                                    save_best_only=True, 
                #                                    save_weights_only=False, 
                #                                    mode='auto', 
                #                                    period=1)
                #                                 ])


                self.Dens_estim.fit([x1_scaled,x2_scaled], y_onehot, epochs=epochs, batch_size=self.batch_size, shuffle=True,
                                        validation_split=0.05,
                                        #validation_data=([input1_t,input2_t], target2_t_onehot),
                                      callbacks=[earlyStopping_1,
                                                keras.callbacks.ModelCheckpoint('weights.hdf5', 
                                                   verbose=0, 
                                                   monitor='val_acc',
                                                   #save_best_only=False, 
                                                   save_best_only=True, 
                                                   save_weights_only=False, 
                                                   mode='auto', 
                                                   period=1)
                                                ])



    def test(self, x_in1, x_in2):

        #print(self.scaler['max'] )
        #print(self.scaler['min'] )
        x1_scaled=x_in1
        x2_scaled=x_in2

        y = self.Dens_estim.predict([x1_scaled,x2_scaled])

        return y
