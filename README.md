## google-isolated-sign-language-recognition
## score at 3rd position is achieved.
![islr_submission](https://github.com/bishnarender/google-isolated-sign-language-recognition/assets/49610834/0153ab28-cecb-4477-bf73-f09d1fb291bd)

-----
### Start 
For better understanding of project, read the files in the following order:
1. eda.ipynb 
2. islr_create_tfrecord.ipynb
3. train.ipynb
4. islr-inference.ipynb

-----
The score achieved when 4 folds are ensembled is 0.8833.
The score achieved when 3 folds are ensembled is 0.8803.
Seeding strategy has not been used i.e., whole data has not been trained on different random seeds.

-----
### islr_create_tfrecord.ipynb
For each video frame (present in parquet file), there are 543 rows. These rows have x,y and z values for various landmarks/keypoints of “Face, Pose, Right hand and Left hand”. So, each parquet file is resized to 3-dimension i.e., (frames, rows_per_frame, 3) or (23,543,3). The 3 present in the last dimension have the value of x,y, and z for respective columns.
The record for 512 parquet files is saved in one tfrecord file.

-----

### train.ipynb
You can watch the actual indices of landmarks/keypoints by zooming the below image.
![indices_landmarks](https://github.com/bishnarender/google-isolated-sign-language-recognition/assets/49610834/27abfba3-c39e-4ccb-848f-c251c97bf9fe)

Function filter_nans_tf(): removes that frame whose all (543*3) values are nan. i.e., if input is (23,543,3) and if 11th frame has all (543*3) as nan then the new input would be (22,543,3).

Augmentaton techniques: Resize, Flip (left to right and vice-versa for various parts of body), Oneof([scale + shear_x + shift + degree, scale + shear_y + shift + degree]), Crop/Masking.

Function resample(): it performs the resize augmentation method. i.e., resizes an input like (23,543,3) to (x, 543, 3) where x is bounded to 50% to 150% of original.

<b><u>How data is standardized prior to feeding:</u></b><br>
Data has been standardized instead of normalization. Now, data has shape in the form [BS,64,543,3]. The first element of the batch is picked i.e., [1,64,543,3]. For this element, first of all the mean is computed across x,y and z position for the lip keypoint/landmark (since it is usually located close to the center ([0.5, 0.5])). Thus, the mean would be of shape [1,1,1,3]. Then standard deviation is computed considering this keypoint/landmark as mean. In this way, all elements of a batch are standardized.

Further, instead of all 543 landmarks/keypoints only the most important ones are selected. Thus, the new form of data would be [BS,64,118,3]. The last value (z coordinate) of the last dimension is discarded and we proceed with [BS,64,118,2]. Two new formats of data are created of the same shape as the original; by applying motion feature lag over 2nd dimension i.e., x[:,1:,:,:] - x[:,:-1,:,:] and x[:,:2,:,:] - x[:,:-2,:,:]. Finally, the three parts are reshaped to [BS, 64, 236] and then all three are concatenated over last dimension to form [BS, 64, 708].
![islr](https://github.com/bishnarender/google-isolated-sign-language-recognition/assets/49610834/3a6eaaad-3a5e-4a0e-a3d0-3eb6bbd39038)

tf.keras.layers.Masking layer has been used to MASK the PAD tokens so that they do not contribute to the calculation of the gradients. As, gradients have not been calculated for PAD tokens so there is no need to add the PAD tokens during the inference, and this will significantly increase the inference time. Maybe someone can recommend to use “PAD=0 and no masking during training” + “no padding during inference”, this strategy reduces accuracy significantly because of the train-test time inconsistency.

TransformerBlock uses 'attention_mask' when training with PAD tokens to make it invisible when training, so it was originally designed to accommodate varied length inputs. Also, masking has been used in the transformer block but that is insufficient.

Causal padding has been used here because if we use typical padding='same', the PAD tokens will influence the original input sequence's time frames, and makes masking unsuccessful.  On the other hand, causal padding always maintains the time frame information while preventing kernels from seeing future frames.
![conv](https://github.com/bishnarender/google-isolated-sign-language-recognition/assets/49610834/69713a93-140c-43c3-888b-a7b3989ad3b7)
