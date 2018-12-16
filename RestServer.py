from flask import Flask
from flask import request
import json
import os
import numpy as np
from trainer.StableFeature import Det_satable
app = Flask(__name__)

 


on_cloud = 0

test_mode=1

epochs=100
batch_size=512*2

encoded_dim=10

data_type = 4
if_real_imag=2
log_path='/Google_drive/Vahid/Dropbox/Working_dir/RestData/StblSec/2dri'
train_prev_model='/Google_drive/Vahid/Dropbox/Working_dir/RestData/StblSec/2dri'
input_shape=(90, 2)
normalize_mode=7


data_type = 4
if_real_imag=2
log_path='/Google_drive/Vahid/Dropbox/Working_dir/RestData/StblSec/2d_turn'
train_prev_model='/Google_drive/Vahid/Dropbox/Working_dir/RestData/StblSec/2d_turn'
input_shape=(90, 2)
normalize_mode=7

data_type = 4
if_real_imag=2
log_path='/Google_drive/Vahid/Dropbox/Working_dir/RestData/StblSec/2d_ext'
train_prev_model='/Google_drive/Vahid/Dropbox/Working_dir/RestData/StblSec/2d_ext'
input_shape=(90, 2)
normalize_mode=7


data_type = 4
if_real_imag=2
log_path='/Google_drive/Vahid/Dropbox/Working_dir/RestData/StblSec/Full_W2'
train_prev_model='/Google_drive/Vahid/Dropbox/Working_dir/RestData/StblSec/Full_W2'
input_shape=(90, 2)
normalize_mode=7


Test_network = Det_satable(on_cloud=on_cloud, encoded_dim=encoded_dim, test_mode =test_mode, just_trained=0 , log_path=log_path, normalize_mode=normalize_mode,
                                data_type=data_type,train_prev_model=train_prev_model, batch_size=batch_size,input_shape=input_shape,if_real_imag=if_real_imag)

import scipy.io
import io
import base64
@app.route("/Stbl_sec", methods = ['POST'])
def Stbl_sec():
    global normalize_mode


@app.route("/Stbl_sec_vjason", methods = ['POST'])
def Stbl_sec_vjason():
  global Test_network
  global normalize_mode
  data = json.loads(request.data.decode('utf-8'))
  #print(data)

  input1r_=np.array(data['input1r'])
  input1i_=np.array(data['input1i'])
  input2r_=np.array(data['input2r'])
  input2i_=np.array(data['input2i'])


  input1=np.zeros((input1r_.shape[1],input1r_.shape[0],2))
  input1[:,:,0]=np.transpose(input1r_)
  input1[:,:,1]=np.transpose(input1i_)

  print(input1.shape)
  print("input 1")
  print(input1[1:10,1:5,0])
  print("input 1- imag")
  print(input1[1:10,1:5,1])

  tmp=input1
  #tmp_mean=np.mean(tmp,axis=1,keepdims=True)
  #tmp=tmp-tmp_mean
  #input1=tmp/np.linalg.norm(tmp,axis=(1),keepdims=True)

  input2=np.zeros((input2r_.shape[1],input2r_.shape[0],2))
  input2[:,:,0]=np.transpose(input2r_)
  input2[:,:,1]=np.transpose(input2i_)
  print("input 2")
  print(input2[1:10,1:5,0])
  print("input 2- imag")
  print(input2[1:10,1:5,1])
  tmp=input2
  #tmp_mean=np.mean(tmp,axis=1,keepdims=True)
  #tmp=tmp-tmp_mean
  #input2=tmp/np.linalg.norm(tmp,axis=(1),keepdims=True)
  print("normalzied")
  print("input 2")
  print(input2[1:10,1:5,0])
  print("input 2- imag")
  print(input2[1:10,1:5,1])


  Y_pred=Test_network.test(input1, input2)

  result = json.dumps(Y_pred.tolist())
  return result

if __name__ == "__main__":
	app.run()	