
# coding: utf-8

# In[1]:

from trainer.StableFeature import Det_satable
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from subprocess import call
import gc


# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import tensorflow as tf
import os
import math
from datetime import datetime
import time
import argparse
#import h5py 
# import cloudstorage as gcs


#from StringIO import StringIO
from tensorflow.python.lib.io import file_io


#on_cloud=1
#data_type=3 # 0: 40K channel, 1: 40K channel and noisy channel at 12db,  1: 40K channel and noisy channel at 22db

# def is_file_available(filepath):
#     try:
#         return gcs.stat(filepath)
#     except gcs_errors.NotFoundError as e:
#         return False

def save_file_toCLoud(filename, Save_loc, save_name):
  print(filename)
  print(Save_loc)
  print(save_name)
  with file_io.FileIO(filename, mode='rb') as input_f:
      with file_io.FileIO(Save_loc+save_name, mode='wb+') as output_f:
          output_f.write(input_f.read())
          print("model/file Saved to GCS")

def norm_data(tmpm):
  Max_v=np.transpose(np.tile(np.max(tmpm,axis=0).reshape(tmpm.shape[1],1),tmpm.shape[0]))   
  Min_v=np.transpose(np.tile(np.min(tmpm,axis=0).reshape(tmpm.shape[1],1),tmpm.shape[0]))   
  tmpm=(tmpm-Min_v)/(Max_v-Min_v) 
  return tmpm


# #matplotlib.use('Qt5Agg')
# def norm_seq(A):
#   max_m=np.max(A,axis=(1,2))+.2
#   min_m=np.min(A,axis=(1,2))-.2
#   min_ma=np.repeat(np.repeat(min_m,72),14).reshape((-1,72,14))
#   max_ma=np.repeat(np.repeat(max_m,72),14).reshape((-1,72,14))
#   return (A-min_ma)/(max_ma-min_ma), max_m, min_m

# def norm_seq_knwon(A,max_m,min_m):
#   min_ma=np.repeat(np.repeat(min_m,72),14).reshape((-1,72,14))
#   max_ma=np.repeat(np.repeat(max_m,72),14).reshape((-1,72,14))
#   return (A-min_ma)/(max_ma-min_ma), max_m, min_m

# def denorm_seq(A, max_m,min_m):
#   min_ma=np.repeat(np.repeat(min_m,72),14).reshape((-1,72,14))
#   max_ma=np.repeat(np.repeat(max_m,72),14).reshape((-1,72,14))
#   return A*(max_ma-min_ma)+min_ma 


def train_model(on_cloud,train_file1,train_file2,job_dir,log_dir="",train_prev_model="",train_mode_on=1,test_mode_on=0,data_type=3):

  # In[7]:
  #def main(_):
  data_path=train_file1
  data_path2=train_file2
  print(on_cloud)

  if (on_cloud == 1):

      log_path = os.path.join(job_dir)
      #log_path = log_path  + datetime.now().isoformat()
      
      data_path  = train_file1
      data_path2 = train_file2

  else:

      log_path = os.path.join(job_dir,log_dir)
      #log_path = log_path  + datetime.now().isoformat()
      

      data_path  = os.path.join(train_file1)
      data_path2 = os.path.join(train_file2)

      print(data_path)

  if data_type==1:
    Data_file1=data_path+"/CSI_Data_ti.mat"

    #Data_file2=data_path2+"/My_noisy_H_12.mat"
  elif data_type==2:
    #Data_file1=data_path+"/CSI_Data_r.mat"
    Data_file1=data_path+"/CSI_Data_r_smoot_ti.mat"
  elif data_type==3:
    #Data_file1=data_path+"/CSI_Data_new_ti_rssi_2.mat"
    Data_file1=data_path+"/CSI_Data_new_ti_rssi_test.mat"
  elif data_type==4:
    #Data_file1=data_path+"/CSI_Data_new_ti_rssi_2.mat"
    Data_file1=data_path+"/CSI_Data_newha_ti_2.mat"
    #Data_file1=data_path+"/CSI_Data_newha_ti_2_short.mat"
    Data_file1=data_path+"/CSI_Data_newha_ti_2_mod_test.mat"


    #Data_file1=data_path+"/My_perfect_H_22.mat"
    #Data_file2=data_path2+"/My_noisy_H_22.mat"

  print(Data_file1)


  #regularizer_coef=0.0000002/1024   
  Train_model=train_mode_on
  Test_model=test_mode_on
  normalize_mode=7 # 1: (a+5)/10, #2: MinMaxScaler, 3: noting 
  encoded_dim=10
  epochs=100
  batch_size=512*2*4
  if_real_imag=2

  if data_type==1 or data_type==2  or data_type==3 or data_type==4:


    print(Data_file1)  
    with file_io.FileIO(Data_file1, mode='rb') as FL:
      if data_type==1 or data_type==2:
          print(FL)
          #Xpca = scipy.io.loadmat(FL)['Xpca']
          input1 = scipy.io.loadmat(FL)['input1']
          input2 = scipy.io.loadmat(FL)['input2']
          target2 = scipy.io.loadmat(FL)['target2']
          input1_t = scipy.io.loadmat(FL)['input1_t']
          input2_t = scipy.io.loadmat(FL)['input2_t']
          target2_t = scipy.io.loadmat(FL)['target2_t']
          input1_ti = scipy.io.loadmat(FL)['input1_ti']
          input2_ti = scipy.io.loadmat(FL)['input2_ti']
          target_ti = scipy.io.loadmat(FL)['target_ti']
      elif data_type==3:
        AA=scipy.io.loadmat(FL,variable_names=['input1', 'input2','input3', 'input4', 'target2', 'input1_t', 'input2_t', 'input3_t', 'input4_t', 'target2_t', 'input1_ti', 'input2_ti', 'input3_ti', 'input4_ti', 'target_ti'])
      elif data_type==4:
        AA=scipy.io.loadmat(FL,variable_names=['input1', 'input2', 'target2', 'input1_t', 'input2_t', 'target2_t', 'input1_ti', 'input2_ti', 'target_ti'])

    def get_part(A):
      return np.absolute(A);

    if normalize_mode==1:
      error()
      # all_channel_noisy_images = (all_channel_noisy_images+5)/10.0
      # all_channel_perfect_images = (all_channel_perfect_images+5)/10.0
    if normalize_mode==5:
      error()
      # all_channel_noisy_images = (all_channel_noisy_images+5)
      # all_channel_perfect_images = (all_channel_perfect_images+5)
    if normalize_mode==6:
      error()
      tmp_,Inputall_max,Inputall_min=norm_seq(inputall)
      inputall,input1_max,input1_min=norm_seq_knwon(inputall,Inputall_max,Inputall_min)

    if normalize_mode==7:
      if data_type==1 or data_type==2:
        inputall = np.vstack([input1, input2])
      elif data_type==3 or data_type==4:
        inputall = np.vstack([AA['input1'], AA['input2']])

    if normalize_mode==8:
      input1=norm_data(input1)
      input2=norm_data(input2)
      input1_ti=norm_data(input1_ti)
      input2_ti=norm_data(input2_ti)
      input1_t=norm_data(input1_t)
      input2_t=norm_data(input2_t)
      inputall = np.vstack([input1, input2])

    if data_type==1 or data_type==2:
      input1=np.transpose(input1)
      input2=np.transpose(input2)
      target2=np.transpose(target2)
      #
      input1_ti=np.transpose(input1_ti)
      input2_ti=np.transpose(input2_ti)
      target_ti=np.transpose(target_ti)
      #Xpca=np.transpose(Xpca)
      input1_t=np.transpose(input1_t)
      input2_t=np.transpose(input2_t)
      target2_t=np.transpose(target2_t)
      inputall=np.transpose(inputall)
    elif data_type==3 or data_type==4:
      if if_real_imag==0:

        input1=np.transpose(get_part(AA['input1']))
        input2=np.transpose(get_part(AA['input2']))
        target2=np.transpose(get_part(AA['target2']))
        #
        input1_ti=np.transpose(get_part(AA['input1_ti']))
        input2_ti=np.transpose(get_part(AA['input2_ti']))
        target_ti=np.transpose(get_part(AA['target_ti']))
        #Xpca=np.transpose(Xpca)
        input1_t=np.transpose(get_part(AA['input1_t']))
        input2_t=np.transpose(get_part(AA['input2_t']))
        target2_t=np.transpose(get_part(AA['target2_t']))

        inputall = np.transpose(get_part(np.vstack([AA['input1'], AA['input2']])))
        #inputall=np.transpose(get_part(inputall))

      if if_real_imag==1:
        input1=np.transpose(np.real(AA['input1']))
        input1=np.concatenate([input1, np.transpose(np.imag(AA['input1']))],axis=1)
        input2=np.transpose(np.real(AA['input2']))
        input2=np.concatenate([input2, np.transpose(np.imag(AA['input2']))],axis=1)
        target2=np.transpose(np.real(AA['target2']))
        #
        input1_ti=np.transpose(np.real(AA['input1_ti']))
        input1_ti=np.concatenate([input1_ti, np.transpose(np.imag(AA['input1_ti']))],axis=1)
        input2_ti=np.transpose(np.real(AA['input2_ti']))
        input2_ti=np.concatenate([input2_ti, np.transpose(np.imag(AA['input2_ti']))],axis=1)
        target_ti=np.transpose(np.real(AA['target_ti']))
        #Xpca=np.transpose(Xpca)
        input1_t=np.transpose(np.real(AA['input1_t']))
        input1_t=np.concatenate([input1_t, np.transpose(np.imag(AA['input1_t']))],axis=1)
        input2_t=np.transpose(np.real(AA['input2_t']))
        input2_t=np.concatenate([input2_t, np.transpose(np.imag(AA['input2_t']))],axis=1)
        target2_t=np.transpose(np.real(AA['target2_t']))
        inputall=np.concatenate([np.transpose(np.real(inputall)), np.transpose(np.imag(inputall))],axis=1)

      if if_real_imag==2:
        input1=np.zeros((AA['input1'].shape[1],AA['input1'].shape[0],2))
        input1[:,:,0]=np.transpose(np.real(AA['input1']))
        input1[:,:,1]=np.transpose(np.imag(AA['input1']))
        #input1[:,:,0]=np.transpose(np.absolute(AA['input1']))
        #input1[:,:,1]=np.transpose(np.absolute(AA['input1']))

        # tmp=input1
        #tmp_mean=np.mean(tmp,axis=1,keepdims=True)
        #tmp=tmp-tmp_mean
        #input1=tmp/np.linalg.norm(tmp,axis=(1),keepdims=True)
        #input1=tmp/np.sqrt(np.var(tmp,axis=(1),keepdims=True))

        print(input1.shape)

        input2=np.zeros((AA['input2'].shape[1],AA['input2'].shape[0],2))
        input2[:,:,0]=np.transpose(np.real(AA['input2']))
        input2[:,:,1]=np.transpose(np.imag(AA['input2']))
        # input2[:,:,0]=np.transpose(np.absolute(AA['input2']))
        # input2[:,:,1]=np.transpose(np.absolute(AA['input2']))

        target2=np.transpose(np.real(AA['target2'][0,:]))

        # tmp=input2
        #tmp_mean=np.mean(tmp,axis=1,keepdims=True)
        #tmp=tmp-tmp_mean
        #input2=tmp/np.linalg.norm(tmp,axis=(1),keepdims=True)
        #input2=tmp/np.sqrt(np.var(tmp,axis=(1),keepdims=True))


        #
        input1_ti=np.zeros((AA['input1_ti'].shape[1],AA['input1_ti'].shape[0],2))
        input1_ti[:,:,0]=np.transpose(np.real(AA['input1_ti']))
        input1_ti[:,:,1]=np.transpose(np.imag(AA['input1_ti']))
        # input1_ti[:,:,0]=np.transpose(np.absolute(AA['input1_ti']))
        # input1_ti[:,:,1]=np.transpose(np.absolute(AA['input1_ti']))

        # tmp=input1_ti
        #tmp_mean=np.mean(tmp,axis=1,keepdims=True)
        #tmp=tmp-tmp_mean
        #input1_ti=tmp/np.linalg.norm(tmp,axis=(1),keepdims=True)
        #input1_t=tmp/np.sqrt(np.var(tmp,axis=(1),keepdims=True))

        input2_ti=np.zeros((AA['input2_ti'].shape[1],AA['input2_ti'].shape[0],2))
        input2_ti[:,:,0]=np.transpose(np.real(AA['input2_ti']))
        input2_ti[:,:,1]=np.transpose(np.imag(AA['input2_ti']))
        # input2_ti[:,:,0]=np.transpose(np.absolute(AA['input2_ti']))
        # input2_ti[:,:,1]=np.transpose(np.absolute(AA['input2_ti']))

        # tmp=input2_ti
        #tmp_mean=np.mean(tmp,axis=1,keepdims=True)
        #tmp=tmp-tmp_mean
        #input2_ti=tmp/np.linalg.norm(tmp,axis=(1),keepdims=True)
        #input2_ti=tmp/np.sqrt(np.var(tmp,axis=(1),keepdims=True))


        target_ti=np.transpose(np.real(AA['target_ti'][0,:]))


        #Xpca=np.transpose(Xpca)
        input1_t=np.zeros((AA['input1_t'].shape[1],AA['input1_t'].shape[0],2))
        input1_t[:,:,0]=np.transpose(np.real(AA['input1_t']))
        input1_t[:,:,1]=np.transpose(np.imag(AA['input1_t']))
        # input1_t[:,:,0]=np.transpose(np.absolute(AA['input1_t']))
        # input1_t[:,:,1]=np.transpose(np.absolute(AA['input1_t']))

        # tmp=input1_t
        #tmp_mean=np.mean(tmp,axis=1,keepdims=True)
        #tmp=tmp-tmp_mean
        #input1_t=tmp/np.linalg.norm(tmp,axis=(1),keepdims=True)
        #input1_t=tmp/np.sqrt(np.var(tmp,axis=(1),keepdims=True))

        input2_t=np.zeros((AA['input2_t'].shape[1],AA['input2_t'].shape[0],2))
        input2_t[:,:,0]=np.transpose(np.real(AA['input2_t']))
        input2_t[:,:,1]=np.transpose(np.imag(AA['input2_t']))
        # input2_t[:,:,0]=np.transpose(np.absolute(AA['input2_t']))
        # input2_t[:,:,1]=np.transpose(np.absolute(AA['input2_t']))

        # tmp=input2_t
        #tmp_mean=np.mean(tmp,axis=1,keepdims=True)
        #tmp=tmp-tmp_mean
        #input2_t=tmp/np.linalg.norm(tmp,axis=(1),keepdims=True)
        #input2_t=tmp/np.sqrt(np.var(tmp,axis=(1),keepdims=True))

        target2_t=np.transpose(np.real(AA['target2_t'][0,:]))

        #inputall_2=np.zeros((inputall.shape[1],AA['input2_t'].shape[0],2))
        inputall = np.concatenate([input1, input2],axis=1)

    AA=None
    gc.collect()


    print(input1.shape)
    print(input2.shape)
    print(inputall.shape)
    print(target2.shape)
    #print(Xpca.shape)

    X_train , X_test, Y_train, Y_test = train_test_split(inputall,target2, test_size=.1, random_state=4000)
    print("----------")
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)

    if data_type==1:

      X_train_1=X_train[:,0:60]
      X_train_2=X_train[:,60:]

      X_test_1=X_test[:,0:60]
      X_test_2=X_test[:,60:]

    elif data_type==2:
      X_train_1=X_train[:,0:120]
      X_train_2=X_train[:,120:]

      X_test_1=X_test[:,0:120]
      X_test_2=X_test[:,120:]

    elif data_type==3 or data_type==4:
      if if_real_imag==0:
        X_train_1=X_train[:,0:90]
        X_train_2=X_train[:,90:]

        X_test_1=X_test[:,0:90]
        X_test_2=X_test[:,90:]
      if if_real_imag==1:
        X_train_1=X_train[:,0:180]
        X_train_2=X_train[:,180:]

        X_test_1=X_test[:,0:180]
        X_test_2=X_test[:,180:]
      if if_real_imag==2:
        X_train_1=X_train[:,0:90,:]
        X_train_2=X_train[:,90:,:]

        X_test_1=X_test[:,0:90,:]
        X_test_2=X_test[:,90:,:]



      Conv_encoder=0
      if Conv_encoder==1:
        X_train_1 = X_train_1.reshape(X_train_1.shape[0], X_train_1.shape[1], 1)
        X_train_2 = X_train_2.reshape(X_train_2.shape[0], X_train_2.shape[1], 1)

        X_test_1 = X_test_1.reshape(X_test_1.shape[0], X_test_1.shape[1], 1)
        X_test_2 = X_test_2.reshape(X_test_2.shape[0], X_test_2.shape[1], 1)

        input1_ti = input1_ti.reshape(input1_ti.shape[0], input1_ti.shape[1], 1)
        input2_ti = input2_ti.reshape(input2_ti.shape[0], input2_ti.shape[1], 1)

        input1_t = input1_t.reshape(input1_t.shape[0], input1_t.shape[1], 1)
        input2_t = input2_t.reshape(input2_t.shape[0], input2_t.shape[1], 1)



  if Train_model==1:
    print('----------')
    print(X_train_1.shape)
    print(X_train_1.shape[1:-1])
    print(X_train_1.shape[1:3])
    if Conv_encoder==0:
      if if_real_imag==0 or if_real_imag==1:
        network = Det_satable(on_cloud=on_cloud, encoded_dim=encoded_dim, test_mode =0, just_trained=0 , log_path=log_path, normalize_mode=normalize_mode,
                                      data_type=data_type,train_prev_model=train_prev_model, batch_size=batch_size,input_shape=X_train_1.shape[1:2],if_real_imag=if_real_imag)
      elif if_real_imag==2:
        network = Det_satable(on_cloud=on_cloud, encoded_dim=encoded_dim, test_mode =0, just_trained=0 , log_path=log_path, normalize_mode=normalize_mode,
                                      data_type=data_type,train_prev_model=train_prev_model, batch_size=batch_size,input_shape=X_train_1.shape[1:3],if_real_imag=if_real_imag)

    if Conv_encoder==1:
      network = Det_satable(on_cloud=on_cloud, encoded_dim=encoded_dim, test_mode =0, just_trained=0 , log_path=log_path, normalize_mode=normalize_mode,
                                    data_type=data_type,train_prev_model=train_prev_model, batch_size=batch_size,input_shape=X_train_1.shape[1:3],if_real_imag=if_real_imag)
    
    if data_type==1 or data_type==2 or data_type==3 or data_type==4:
      network.train(X_train_1, X_train_2, Y_train,input1_t, input2_t,target2_t, epochs=epochs, batch_size=batch_size)



  if Test_model==1:
    # if (Test_model==1) and (on_cloud==1):
    #   arg="gsutil cp "+job_dir+"/weights.hdf5 gs://cloudchest/Prev_model/weights.hdf5"
    #   print(arg)
    #   call([arg])
    #   arg="gsutil cp "+job_dir+"/weights_1.hdf5 gs://cloudchest/Prev_model/weights_1.hdf5"
    #   call([arg])
    if Conv_encoder==0:
      if if_real_imag==0 or if_real_imag==1:
        testnetwork = Det_satable(on_cloud=on_cloud, encoded_dim=encoded_dim, test_mode =1, just_trained=1 , log_path=log_path, normalize_mode=normalize_mode,
                                      data_type=data_type,train_prev_model=train_prev_model, batch_size=batch_size,input_shape=X_train_1.shape[1:2],if_real_imag=if_real_imag)
      elif if_real_imag==2:
        testnetwork = Det_satable(on_cloud=on_cloud, encoded_dim=encoded_dim, test_mode =1, just_trained=1 , log_path=log_path, normalize_mode=normalize_mode,
                                      data_type=data_type,train_prev_model=train_prev_model, batch_size=batch_size,input_shape=X_train_1.shape[1:3],if_real_imag=if_real_imag)
    if Conv_encoder==1:
      testnetwork = Det_satable(on_cloud=on_cloud, encoded_dim=encoded_dim, test_mode =1, just_trained=1 , log_path=log_path, normalize_mode=normalize_mode,
                                    data_type=data_type,train_prev_model=train_prev_model, batch_size=batch_size,input_shape=X_train_1.shape[1:3],if_real_imag=if_real_imag)

    if data_type==1 or data_type==2 or data_type==3 or data_type==4:
      Y_pred=testnetwork.test(X_test_1, X_test_2)
      print("I am here")
      print(Y_pred.shape)
      print(Y_test.shape)
      # if data_type==3 and if_real_imag==2:
      #   Y_s=np.concatenate((Y_pred[:,0],Y_test))
      # else:
      #   Y_s=np.concatenate((Y_pred,Y_test),axis=1)

      # Test_error=np.mean(np.square(Y_test-Y_pred))
      # print(Test_error)
      np.savetxt("Y_pred_itself.csv", Y_pred, delimiter=",")
      np.savetxt("Y_real_itself.csv", Y_test, delimiter=",")

      same_group_pred=(Y_pred[:,0]>0.5)
      if (data_type==3 or data_type==4) and if_real_imag==2:
        same_group_real=(Y_test>0.5)
      else:
        same_group_real=(Y_test[:,0]>0.5)

      In_same_group=np.sum(same_group_real)
      Not_In_same_group=np.sum(1-same_group_real)

      Test_error=np.mean((same_group_pred^same_group_real))
      print(Test_error)

      same_group_but_error=np.sum((1-same_group_pred)*same_group_real)
      not_same_group_and_error=np.sum(same_group_pred*(1-same_group_real))
      same_group_and_correct=np.sum(same_group_pred*same_group_real)
      not_same_group_and_correct=np.sum((1-same_group_pred)*(1-same_group_real))

      print("same_group_and_error: "+str(same_group_but_error)+"/"+str(In_same_group))
      print("not_same_group_and_error: "+str(not_same_group_and_error)+"/"+str(Not_In_same_group))
      print("same_group_and_correct: "+str(same_group_and_correct)+"/"+str(In_same_group))
      print("not_same_group_and_correct: "+str(not_same_group_and_correct)+"/"+str(Not_In_same_group))
      print("----------------------------------------------")
      print("In_same_group: "+str(In_same_group))
      print("Not_In_same_group: "+str(Not_In_same_group))


      print('==================================================')
      Y_pred=testnetwork.test(input1_ti, input2_ti)
      Y_test=target_ti;

      # if data_type==3 and if_real_imag==2:
      #   Y_s=np.concatenate((Y_pred[:,0],Y_test),axis=0)
      # else:
      #   Y_s=np.concatenate((Y_pred,Y_test),axis=1)

      np.savetxt("Y_pred_inner.csv", Y_pred, delimiter=",")
      np.savetxt("Y_real_inner.csv", Y_test, delimiter=",")


      same_group_pred=(Y_pred[:,0]>0.5)
      if (data_type==3 or data_type==4) and if_real_imag==2:
        same_group_real=(Y_test>0.5)
      else:
        same_group_real=(Y_test[:,0]>0.5)
      In_same_group=np.sum(same_group_real)
      Not_In_same_group=np.sum(1-same_group_real)

      # Test_error=np.mean(np.square(Y_test-Y_pred))
      # print(Test_error)
      Test_error=np.mean((same_group_pred^same_group_real))
      print(Test_error)

      same_group_but_error=np.sum((1-same_group_pred)*same_group_real)
      not_same_group_and_error=np.sum(same_group_pred*(1-same_group_real))
      same_group_and_correct=np.sum(same_group_pred*same_group_real)
      not_same_group_and_correct=np.sum((1-same_group_pred)*(1-same_group_real))

      print("same_group_and_error: "+str(same_group_but_error)+"/"+str(In_same_group))
      print("not_same_group_and_error: "+str(not_same_group_and_error)+"/"+str(Not_In_same_group))
      print("same_group_and_correct: "+str(same_group_and_correct)+"/"+str(In_same_group))
      print("not_same_group_and_correct: "+str(not_same_group_and_correct)+"/"+str(Not_In_same_group))
      print("----------------------------------------------")
      print("In_same_group: "+str(In_same_group))
      print("Not_In_same_group: "+str(Not_In_same_group))

      print('==================================================')
      Y_pred=testnetwork.test(input1_t, input2_t)
      Y_test=target2_t;

      # Y_s=np.concatenate((Y_pred[:,0],Y_test),axis=1)

      np.savetxt("Y_pred_other.csv", Y_pred, delimiter=",")
      np.savetxt("Y_real_other.csv", Y_test, delimiter=",")

      same_group_pred=(Y_pred[:,0]>0.5)
      if (data_type==3 or data_type==4) and if_real_imag==2:
        same_group_real=(Y_test>0.5)
      else:
        same_group_real=(Y_test[:,0]>0.5)

      Test_error=np.mean((same_group_pred^same_group_real))
      print(Test_error)

      In_same_group=np.sum(same_group_real)
      Not_In_same_group=np.sum(1-same_group_real)

      same_group_but_error=np.sum((1-same_group_pred)*same_group_real)
      not_same_group_and_error=np.sum(same_group_pred*(1-same_group_real))
      same_group_and_correct=np.sum(same_group_pred*same_group_real)
      not_same_group_and_correct=np.sum((1-same_group_pred)*(1-same_group_real))

      print("same_group_and_error: "+str(same_group_but_error)+"/"+str(In_same_group))
      print("not_same_group_and_error: "+str(not_same_group_and_error)+"/"+str(Not_In_same_group))
      print("same_group_and_correct: "+str(same_group_and_correct)+"/"+str(In_same_group))
      print("not_same_group_and_correct: "+str(not_same_group_and_correct)+"/"+str(Not_In_same_group))
      print("----------------------------------------------")
      print("In_same_group: "+str(In_same_group))
      print("Not_In_same_group: "+str(Not_In_same_group))

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  # Input Arguments
  parser.add_argument(
    '--on-cloud',
    help='If on Cloud:',
    required=True
  )

  parser.add_argument(
    '--train-file1',
    help='GCS or local paths to training data',
    required=True
  )
  parser.add_argument(
    '--train-file2',
    help='GCS or local paths to training data',
    required=True
  )

  # if (on_cloud == 1):
  #     print("on cloud")
  # else:
  #     parser.add_argument(
  #     '--log-dir',
  #     help='GCS or local paths to save results',
  #     required=True
  #     )

  parser.add_argument(
    '--job-dir',
    help='GCS location to write checkpoints and export models',
    required=True
  )
  
  parser.add_argument(
    '--train-prev-model',
    help='GCS location to write checkpoints and export models',
    required=True
  )
  parser.add_argument(
    '--train-mode-on',
    help='GCS location to write checkpoints and export models',
    required=True
  )
  parser.add_argument(
    '--test-mode-on',
    help='GCS location to write checkpoints and export models',
    required=True
  )

  parser.add_argument(
    '--data-type',
    help='GCS location to write checkpoints and export models',
    required=True
  )
  
  args = parser.parse_args()
  arguments = args.__dict__
  on_cloud = int(float(arguments.pop('on_cloud')))
  train_file1 = arguments.pop('train_file1')
  train_file2 = arguments.pop('train_file2')
  job_dir = arguments.pop('job_dir')
  train_prev_model = arguments.pop('train_prev_model')
  train_mode_on = int(float(arguments.pop('train_mode_on')))
  test_mode_on = int(float(arguments.pop('test_mode_on')))
  data_type = int(float(arguments.pop('data_type')))
  if (on_cloud == 1):
    log_dir = ""
  else:
    log_dir = "Logs"
  
  print(log_dir)
  
  train_model(on_cloud,train_file1,train_file2,job_dir,log_dir,train_prev_model,train_mode_on,test_mode_on,data_type)

