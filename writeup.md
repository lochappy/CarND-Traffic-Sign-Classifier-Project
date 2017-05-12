# **Traffic Sign Recognition** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/statisTrainData.png "Visualization"
[image2]: ./images/nomalizedImage.png "Normalized Image"
[image3]: ./images/augmentedData.png "Augmented Image"
[image4]: ./5_samples/1.jpg "Traffic Sign 1"
[image5]: ./5_samples/2.jpg "Traffic Sign 2"
[image6]: ./5_samples/3.jpg "Traffic Sign 3"
[image7]: ./5_samples/4.jpg "Traffic Sign 4"
[image8]: ./5_samples/5.jpg "Traffic Sign 5"
[image9]: ./images/confidentScores.png "confidentScores"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/lochappy/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the normal python library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third and fourth code cells of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distribute among the class labels. The training data are quite skewed. Some classes have high number of training samples such as 1,2,12,38, etc., while the others have much more smaller one like 0, 19, 32, etc.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the 5th code cell of the IPython notebook.

As a first step, I decided to euqualize the histogram of the images using [Contrast Limited Adaptive Histogram Equalization](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html) because I realized that the most of the images were taken at very low contrast condition.

Here is an example of a traffic sign image before and after normalization.

![alt text][image2]

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

As the dataset has provided the validation and test set, I just used them without splitting the training data further. Instead, I focused more on augmenting to the training data.

The sixth code cell of the IPython notebook contains the code for augmenting the data set. My augmentation followed the one suggested in [this paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). Specifically, the data was randomly rotated in [-16,16] degrees, shifted in [-5,5] pixels, and scaled in [0.8,1.2]. In addition, the color of images was also distorted with RGB values in [-10,10] magnitude. The augmentation was performed 10 times in the original data, making the 10 times increasment of the number of training data.

Here is an example of an original image and an augmented image:

![alt text][image3]

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 9th cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Conv1 5x5     	    | 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Conv2 5x5     	    | 1x1 stride, valid padding, outputs 10x10x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x24 				    |
| Flatten1	            | outputs 600     								|
| Conv3 3x3     	    | 1x1 stride, valid padding, outputs 3x3x48 	|
| RELU					|												|
| Flatten2	            | outputs 432                                   |
| Concat(Flatten1 + Flatten2) | outputs 1032                            |
| Fully connected 1	    | outputs 512       							|
| Fully connected 2	    | outputs 256       							|
| Fully connected 3	    | outputs 43       							    |
| Softmax				| outputs 43             					    |
 


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 15th cell of the ipython notebook. 

To train the model, I used an Adam optimizer with learning rate of 1e-1 as stated in 13rd cell. The model was trained across 1000 epochs with batch size of 32. The best accuracy recorded at epoch 380th was 99.1% on the original validation set.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 12th cell of the Ipython notebook. Here is my iterative process to find a suitable architecture:
1. Design a architecture
2. Train model
3. Test on validation set, if the accuracy is less than requirement, modify the architecture, then go to step 2.

My final model results were:
* validation set accuracy of 99.1%

My architecture was similar to the one described in [this paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), which exploited the multiscale feature maps of the CNN to improve the overall accuracy of the models. Remarkably, instead of using dropout, which actually harmed the perfomance, this architecture used the skip connection, which connected the feature map of the second convolutional layer directly to the 4th one. As a result, this connection let the information of smaller scale, which could vanish through the pooling layer, flow directly to the fully connected layer, making the model more scale-invariant.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web. The quality of these images seems to be quite good to human perception. Let's try this on the train model.

| Image			    |Ground truth       |Brightness|Contrast|Noise   |
|:-----------------:|:-----------------:|:--------:|:------:|:------:|
|![alt text][image8]|Children Crossing	| High     |Good    |Tree branches in the backgound, watermark on the sign|
|![alt text][image5]|Pedestrians 	    | High     |Good    |Tree branches in the backgound, watermark on the sign|
|![alt text][image6]|Runabout Mandantory| High     |Good    |Watermark on the sign|
|![alt text][image4]|Speed limit 60 km/h| High     |Good    |Watermark on the sign with perspective distortion|
|![alt text][image7]|Keep right      	| High     |Good    |Watermark on the sign with perspective distortion|


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 19th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			    | Ground truth			|     Prediction	        					| 
|:-----------------:|:---------------------:|:---------------------------------------------:| 
|![alt text][image8]| Children Crossing     | Children Crossing   							| 
|![alt text][image5]| Pedestrians     		| Pedestrians 									|
|![alt text][image6]| Runabout Mandantory	| Runabout Mandantory							|
|![alt text][image4]| Speed limit 60 km/h	| Speed limit 60 km/h					 		|
|![alt text][image7]| Keep right			| Keep right      							    |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the validation set of 99.1%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 20th cell of the Ipython notebook.

For all of the images in the new test set, the model was quite certain in predicting their labels. Specifically, the confident score of each sample was almost 1 as shown in the following bar chart.

![alt text][image9]

