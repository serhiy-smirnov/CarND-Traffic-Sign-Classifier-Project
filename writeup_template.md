# **Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[dataset_stats]: ./images/SignsDistribution.png "Dataset Visualization"
[dataset_images]: ./images/ImagesRandomSnapshot.jpg "Dataset Visualization"
[overfitting1]: ./images/overfitting1.jpg "Overfitting 1"
[overfitting2]: ./images/overfitting2.jpg "Overfitting 2"
[overfitting3]: ./images/overfitting3.jpg "Overfitting 3"
[overfitting4]: ./images/overfitting4.jpg "Overfitting 4"
[overfitting5]: ./images/overfitting5.jpg "Overfitting 5"
[overfitting6]: ./images/overfitting6.jpg "Overfitting 6"
[sign1]: ./signs/1.jpg "Traffic Sign 1"
[sign2]: ./signs/2.jpg "Traffic Sign 2"
[sign3]: ./signs/3.jpg "Traffic Sign 3"
[sign4]: ./signs/4.jpg "Traffic Sign 4"
[sign5]: ./signs/5.jpg "Traffic Sign 5"
[sign6]: ./signs/6.jpg "Traffic Sign 6"
[sign7]: ./signs/7.jpg "Traffic Sign 7"
[sign8]: ./signs/8.jpg "Traffic Sign 8"
[sign9]: ./signs/9.jpg "Traffic Sign 9"
[sign10]: ./signs/10.jpg "Traffic Sign 10"
[layers_visualization]: ./images/layers_visualization.jpg "Layers visualization"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it!

Here is a link to my [project code](https://github.com/serhiy-smirnov/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

iPython Notebook exported with run-time results can be found [here](https://github.com/serhiy-smirnov/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.

It is a bar chart showing how the data in the training/validation/test datasets is distributed across unique classes:

![alt text][dataset_stats]

And here're some examples of the images used in the dataset:

![alt text][dataset_images]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I normalized the image data so that it has mean zero and equal variance.
After evaluating initial validation and testing accuracy I decided to skip additional pre-processing steps and focused on the model configuration and tuning as well as extra task of visualizing the network layers output (see below).


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Flatten	      	    | outputs 400                				    |
| Fully connected		| outputs 120  									|
| RELU					|												|
| Dropout				| with keep probability 0.5 for training        |
| Fully connected		| outputs 84  									|
| RELU					|												|
| Fully connected		| outputs n_classes (43)  				        |
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Before I came to this final model, I went through several iterations to find an optimal configuration and parameters that allowed me to achieve target accuracy level:

1. No dropout. Run with 200 epochs. Clear overfitting on the training dataset. Result doesn't go higher than ~0.9 with validation and test sets.
![alt text][overfitting1]

2. Added dropout after both convolutional layers with initial keep parameter of 0.5. Run with 25 epochs. Result is about 0.8 on the validation set.

3. Increased number of epochs to 100. Validation accuracy reaches 0.91, test accuracy still below 0.9
![alt text][overfitting2]

4. Removed dropout after c1 and c3, added dropout after first fully connected layer (c5). Testing result improved up to 0.94
![alt text][overfitting3]

5. Added another dropout after second fully connected layer (f6). Testing result dropped down to 0.91
![alt text][overfitting4]

6. Removed first dropout after c5. Run with dropout after f6. Testing result improved back to 0.934
![alt text][overfitting5]

7. Back to the drop out after c5. Implemented early stop mechanism with parameter of epochs number over which validation accuracy was not improving (started with 10). Early stop occured after epoch 62 with testing result of 0.933
![alt text][overfitting6]

8. After implementation of data normalization, early stop kicks in after ~43 epochs allowing to reach target validation accuracy of more than 0.93

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.956
* test set accuracy of 0.942
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I have chosen 10 German traffic signs, including duplicates of the same sign class:

![alt text][sign1]{height=50px width=50} ![alt text][sign2] ![alt text][sign3] ![alt text][sign4] ![alt text][sign5]
![alt text][sign6] ![alt text][sign7] ![alt text][sign8] ![alt text][sign9] ![alt text][sign10]

The second and the fifth images might be difficult to classify because the signs are rotated in relation to the camera therefore their shapes are distorted. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield      		    | Yield        									| 
| Slippery road			| Slippery road									|
| Turn right ahead		| Ahead only									|
| Roundabout mandatory  | Roundabout mandatory                          |
| No entry			    | No entry            							|
| Speed limit (30km/h)  | Speed limit (20km/h)							|
| No entry			    | No entry            							|
| Speed limit (100km/h) | Speed limit (30km/h) 							|
| Road narrows on the right| Road narrows on the right                  |
| Stop			        | Stop              							|


The model was able to correctly guess 7 of the 10 traffic signs, which gives an accuracy of 70%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

Top 5 probabilities for each detection are shown below.

```
Sign file './signs\1.jpg' (reference label '13' - 'Yield') is detected with the following probabilities:
 with probability of 1.00000 as sign class '13' - 'Yield'
 with probability of 0.00000 as sign class '9' - 'No passing'
 with probability of 0.00000 as sign class '3' - 'Speed limit (60km/h)'
 with probability of 0.00000 as sign class '35' - 'Ahead only'
 with probability of 0.00000 as sign class '15' - 'No vehicles'

Sign file './signs\10.jpg' (reference label '23' - 'Slippery road') is detected with the following probabilities:
 with probability of 1.00000 as sign class '23' - 'Slippery road'
 with probability of 0.00000 as sign class '19' - 'Dangerous curve to the left'
 with probability of 0.00000 as sign class '10' - 'No passing for vehicles over 3.5 metric tons'
 with probability of 0.00000 as sign class '9' - 'No passing'
 with probability of 0.00000 as sign class '30' - 'Beware of ice/snow'

Sign file './signs\2.jpg' (reference label '33' - 'Turn right ahead') is detected with the following probabilities:
 with probability of 1.00000 as sign class '35' - 'Ahead only'
 with probability of 0.00000 as sign class '34' - 'Turn left ahead'
 with probability of 0.00000 as sign class '3' - 'Speed limit (60km/h)'
 with probability of 0.00000 as sign class '33' - 'Turn right ahead'
 with probability of 0.00000 as sign class '40' - 'Roundabout mandatory'

Sign file './signs\3.jpg' (reference label '40' - 'Roundabout mandatory') is detected with the following probabilities:
 with probability of 1.00000 as sign class '40' - 'Roundabout mandatory'
 with probability of 0.00000 as sign class '32' - 'End of all speed and passing limits'
 with probability of 0.00000 as sign class '38' - 'Keep right'
 with probability of 0.00000 as sign class '7' - 'Speed limit (100km/h)'
 with probability of 0.00000 as sign class '6' - 'End of speed limit (80km/h)'

Sign file './signs\4.jpg' (reference label '17' - 'No entry') is detected with the following probabilities:
 with probability of 1.00000 as sign class '17' - 'No entry'
 with probability of 0.00000 as sign class '14' - 'Stop'
 with probability of 0.00000 as sign class '9' - 'No passing'
 with probability of 0.00000 as sign class '29' - 'Bicycles crossing'
 with probability of 0.00000 as sign class '28' - 'Children crossing'

Sign file './signs\5.jpg' (reference label '1' - 'Speed limit (30km/h)') is detected with the following probabilities:
 with probability of 0.81405 as sign class '0' - 'Speed limit (20km/h)'
 with probability of 0.14494 as sign class '29' - 'Bicycles crossing'
 with probability of 0.02016 as sign class '1' - 'Speed limit (30km/h)'
 with probability of 0.01938 as sign class '28' - 'Children crossing'
 with probability of 0.00128 as sign class '5' - 'Speed limit (80km/h)'

Sign file './signs\6.jpg' (reference label '17' - 'No entry') is detected with the following probabilities:
 with probability of 1.00000 as sign class '17' - 'No entry'
 with probability of 0.00000 as sign class '14' - 'Stop'
 with probability of 0.00000 as sign class '29' - 'Bicycles crossing'
 with probability of 0.00000 as sign class '12' - 'Priority road'
 with probability of 0.00000 as sign class '0' - 'Speed limit (20km/h)'

Sign file './signs\7.jpg' (reference label '7' - 'Speed limit (100km/h)') is detected with the following probabilities:
 with probability of 0.97757 as sign class '1' - 'Speed limit (30km/h)'
 with probability of 0.02239 as sign class '5' - 'Speed limit (80km/h)'
 with probability of 0.00003 as sign class '0' - 'Speed limit (20km/h)'
 with probability of 0.00001 as sign class '6' - 'End of speed limit (80km/h)'
 with probability of 0.00000 as sign class '16' - 'Vehicles over 3.5 metric tons prohibited'

Sign file './signs\8.jpg' (reference label '24' - 'Road narrows on the right') is detected with the following probabilities:
 with probability of 0.99998 as sign class '24' - 'Road narrows on the right'
 with probability of 0.00001 as sign class '25' - 'Road work'
 with probability of 0.00000 as sign class '18' - 'General caution'
 with probability of 0.00000 as sign class '26' - 'Traffic signals'
 with probability of 0.00000 as sign class '27' - 'Pedestrians'

Sign file './signs\9.jpg' (reference label '14' - 'Stop') is detected with the following probabilities:
 with probability of 1.00000 as sign class '14' - 'Stop'
 with probability of 0.00000 as sign class '29' - 'Bicycles crossing'
 with probability of 0.00000 as sign class '17' - 'No entry'
 with probability of 0.00000 as sign class '15' - 'No vehicles'
 with probability of 0.00000 as sign class '18' - 'General caution'
```

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Visualization of different network layers can be seen below:

![alt text][layers_visualization]

