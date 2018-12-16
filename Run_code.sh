
export Test_case="v_ext_drop0"
#export prev_trained="drop10"
export prev_trained="empty"

export BUCKET_NAME=stablech
#export JOB_NAME="chest_$(date +%Y%m%d_%H%M%S)"
export JOB_NAME="re_Stable_$Test_case"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=us-central1


gcloud ml-engine jobs submit training $JOB_NAME \
  --job-dir gs://$BUCKET_NAME/$JOB_NAME \
  --module-name trainer.main \
  --package-path ./trainer \
  --region $REGION \
  --config=trainer/cloudml-gpu.yaml \
  --runtime-version 1.4 \
  -- \
  --on-cloud=1 \
  --train-mode-on=1 \
  --test-mode-on=1 \
  --data-type=4 \
  --train-file1 gs://$BUCKET_NAME/Channel_turn \
  --train-file2 gs://$BUCKET_NAME/Channel_turn \
  --train-prev-model gs://$BUCKET_NAME/Prev_model/$prev_trained

gcloud ml-engine jobs stream-logs $JOB_NAME

gcloud ml-engine jobs stream-logs  re_Stable_v_ext_drop0



#copy data from laptop to VPS
pscp CSI_Data.mat vahid@vps174888.vps.ovh.ca:/home/vahid/Gcloud/Data/Channel_turn
pscp CSI_Data_newha_ti_2.mat vahid@vps174888.vps.ovh.ca:/home/vahid/Gcloud/Data/Channel_turn
pscp CSI_Data_newha_ti_2.mat ali@172.21.242.240:/home/ali/vahid/stblcsi/Data
pscp * ali@172.21.242.240:/home/ali/vahid/chmeas/reconst_48_12

#copy data from vps to cloud for data:
gsutil -m cp * gs://stablech/Channel_turn

=================================================

#copy code from laptop to VPS
pscp -r *.* vahid@vps174888.vps.ovh.ca:/home/vahid/Gcloud/2ri_Stable_GCL
pscp -r *.* ali@172.21.242.240:/home/ali/vahid/stblcsi/2ri_Stable_GCL
pscp -r *.* ali@217.219.236.208:/home/ali/vahid/stblcsi/2ri_Stable_GCL

#run code on GCL from VPS (Puttu)
The above code

#copy from results in cloud to Prev_model folder in cloud
gsutil cp gs://stablech/$JOB_NAME/weights.hdf5 gs://stablech/Prev_model/weights.hdf5
gsutil cp gs://stablech/re_Stable__v4/weights.hdf5 gs://stablech/Prev_model/drop10/weights.hdf5
gsutil cp gs://stablech/re_Stable_v_ext_drop0/weights.hdf5 .


#copy weights from VPS folder to GCL
gsutil cp weights.hdf5 gs://stablech/Prev_model/weights.hdf5


#copy from Prev_model folder in cloud into Trained folder on VPS
gsutil -m cp gs://stablech/Prev_model/* /home/vahid/Gcloud/Trained_Stable

#copy from Trained folder on VPS into laptop
export Resulted_folder="/Dropbox/Working_dir/Tensorflow_home/Share_weights/48_flex_norm7_conv"
pscp -r vahid@vps174888.vps.ovh.ca:/home/vahid/Gcloud/Trained/* $Resulted_folder

pscp -r vahid@vps174888.vps.ovh.ca:/home/vahid/Gcloud/Trained/* /Dropbox/Working_dir/Tensorflow_home/Share_weights/36_flex_norm7_1

pscp -r vahid@vps174888.vps.ovh.ca:/home/vahid/Gcloud/Trained_Stable/* .
#test on Laptop 
python -m trainer.main --on-cloud=0   --train-file1=C:/Local_data/working_large/Stable_ch_measurment --train-file2=/Local_data/working_large/Stable_ch_measurment --job-dir=. --train-prev-model=. --train-mode-on=1 --test-mode-on=1 --data-type=4
python -m trainer.main --on-cloud=0   --train-file1=C:/Local_data/working_large/Stable_ch_measurment --train-file2=/Local_data/working_large/Stable_ch_measurment --job-dir=. --train-prev-model=. --train-mode-on=0 --test-mode-on=1 --data-type=4

gsutil -m cp weights.hdf5 gs://stablech/Prev_model/


python -m trainer.main --on-cloud=0   --train-file1=../Data --train-file2=../Data --job-dir=. --train-prev-model=. --train-mode-on=1 --test-mode-on=1 --data-type=4
python -m trainer.main --on-cloud=0   --train-file1=../Data --train-file2=../Data --job-dir=. --train-prev-model=. --train-mode-on=0 --test-mode-on=1 --data-type=4
