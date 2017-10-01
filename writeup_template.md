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

[image1]: ./subset_train_images.png "Random traffic sign visualization"
[image2]: ./traffic_sign_number.png "Traffic sign number"
[image3]: ./original_image.png "Original image"
[image4]: ./yuv_image.png "YUV colorspace (Y channel)"
[image5]: ./global_contrast_image.png "After global contrast"
[image6]: ./local_contrast_image.png "After local contrast"
[image7]: ./rotated.png "Rotated image"
[image8]: ./zoomed.png "Zoomed image"
[image9]: ./rotated_zoomed.png "Rotated and zoomed image"
[image10]: ./new-images/100_1607.jpg
[image11]: ./new-images/459381023.jpg
[image12]: ./new-images/459381063.jpg
[image13]: ./new-images/459381091.jpg
[image14]: ./new-images/459381273.jpg
[image15]: ./new-images/465921901.jpg
[image16]: ./new-images/german-traffic-sign-caution-roadworks-71151565.jpg
[image17]: ./new-images/index.jpeg
[image18]: ./new-images/traffic-sign-stock-picture-199869.jpg
[image19]: ./cnn_filters.png

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/hvillanua/CarND-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the amount of traffic signs per class.

![alt text][image1]
![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to implement the architecture propsed in this [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)

As a first step, I convert the images to YUV colorspace so I have access to a luminance channel without losing information about color.

Here is an example of a traffic sign image before and after changing to YUV colorspace.

![alt text][image3]
![alt text][image4]

As a last step, I apply global normalization followd by local contrast normalization to every channel independently of the others.

Here is an example of a traffic sign image after global and local contrast normalization.

![alt text][image5]
![alt text][image6]

NOTE: I also tried applying histogram equalization (both regular and adaptative), but the results were worse than the current setup.

I decided to generate additional data because some of the traffic signs had less data and this made the performance of the classification
go down on these images.

To add more data to the the data set, I applied rotation (-15º to 15º), zoom (0.9 to 1.1) and both at the same time, as suggested in the paper.
Data augmentation made the final model more robust, increasing the validation accuracy by ~3%.

Here is an example of three augmented images produced:

![alt text][image7]
![alt text][image8]
![alt text][image9]

The difference between the original data set and the augmented data set is that certain classes with a number of images below a certain threshold
now obtain more images to train on.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 YUV image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x108. Applies only to Y channel. 	|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x8. Applies only to UV channels. 	|
| Concatenation     	| Concatenate filter generated by previous conv layers. 	|
| RELU					|												|
| Batch normalization					|												|
| Max pooling 5x5	      	| 2x2 stride, same padding, outputs 16x16x116 				|
| Max pooling 5x5	| Branched pooling from previous max pool layer. 2x2 stride, same padding, outputs 8x8x116.     									|
| Convolution 5x5	| Uses as input the output of the first max pooling layer, not the branched one. 1x1 stride, same padding, outputs 16x16x108.      									|
| RELU					|												|
| Batch normalization					|												|
| Max pooling 5x5	      	| 2x2 stride, same padding, outputs 8x8x108 				|
| Concatenation     	| Concatenate flattened filters generated by previous layer and the branched one. 	|
| Fully connected		| 100 neurons.        									|
| Fully connected		| 43 neurons (1 per class).        									|
| Softmax and cross-entropy				|         									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer. After tweaking the parameters I ended up getting the best results with this setup:
* YUV images
* learning_rate = 0.00002. This is the highest I could go without making the model stall after a few epochs.
* epochs = 200. Around epoch 200 seems to stop improving. Haven't tried going for 300.
* batch_size = 512. Tried several lower values. 512 gave the best results.

When calculating the accuracy over the validation set while training I had to drop the last 500 images from the validation set since
I would run on Out Of Memory (OOM) error. After training, the accuracy on the whole validation set was actually a little bit higher
than the one reported at the last epoch.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.6%
* validation set accuracy of 96.1%
* test set accuracy of 92.8%

|	Traffic Sign	| Precision	|    recall	|	f1-score	|	support		|
|:--------------:|:--------------:|:---------------------:|:----------:|:----------:|
|Speed limit (20km/h)    |   0.83   |   0.98   |   0.90    |    60	|
|Speed limit (30km/h)    |   0.99   |   0.92   |   0.95    |   720	|
|Speed limit (50km/h)    |   0.93   |   0.99   |   0.96    |   750	|
|Speed limit (60km/h)    |   0.87   |   0.91   |   0.89    |   450	|
|Speed limit (70km/h)    |   0.97   |   0.96   |   0.96    |   660	|
|Speed limit (80km/h)    |   0.89   |   0.90   |   0.90    |   630	|
|End of speed limit (80km/h)    |   0.91   |   0.91   |   0.91    |   150	|
|Speed limit (100km/h)   |    0.92   |   0.89   |   0.90    |   450	|
|Speed limit (120km/h)    |   0.96   |   0.88   |   0.92    |    450		|
|No passing    |   0.94   |   0.96   |   0.95    |   480		|
|No passing for vehicles over 3.5 metric tons   |    0.98   |   0.97   |   0.98   |    660	|
|Right-of-way at the next intersection   |    0.92   |   0.86   |   0.89    |   420	|
|Priority road    |   0.98   |   0.97   |   0.97    |   690	|
|Yield   |    0.97  |   1.00  |    0.98   |    720	|
|Stop    |   0.98   |   1.00  |    0.99   |    270	|
|No vehicles    |   0.91   |   0.99   |   0.95   |    210	|
|Vehicles over 3.5 metric tons prohibited   |    0.96   |   1.00   |   0.98   |    150	|
|No entry    |   1.00   |   0.98   |   0.99    |   360	|
|General caution   |   0.95   |   0.90   |   0.92    |   390	|
|Dangerous curve to the left    |   0.84   |   0.98   |   0.91    |    60	|
|Dangerous curve to the right   |    0.64   |   0.86   |   0.73    |    90	|
|Double curve   |    0.90   |   0.59   |   0.71   |     90	|
|Bumpy road    |   0.82   |   0.93   |   0.87    |   120	|
|Slippery road   |    0.83   |   0.87   |   0.85   |    150	|
|Road narrows on the right    |   0.98  |    0.72   |   0.83    |    90	|
|Road work    |   0.93   |   0.90   |   0.92   |    480	|
|Traffic signals   |    0.79   |   0.88   |   0.83   |    180	|
|Pedestrians    |   0.96   |   0.45   |   0.61     |   60	|
|Children crossing   |    0.66   |   0.93   |   0.77   |    150	|
|Bicycles crossing   |    0.66   |   0.74   |   0.70   |     90	|
|Beware of ice/snow    |   0.69   |   0.68   |   0.68   |    150	|
|Wild animals crossing   |    0.97   |   0.97   |   0.97    |   270	|
|End of all speed and passing limits   |    0.87   |   1.00   |   0.93   |     60	|
|Turn right ahead   |    0.97   |   0.99   |   0.98   |    210	|
|Turn left ahead    |   0.95    |  0.98   |   0.97    |   120	|
|Ahead only   |    0.96   |   0.89   |   0.93   |    390	|
|Go straight or right   |    0.93   |   0.97   |   0.95   |    120	|
|Go straight or left    |   0.95    |  0.98   |   0.97    |    60	|
|Keep right   |    0.99   |   0.94   |   0.97    |   690	|
|Keep left   |    0.93   |   0.94   |   0.94    |    90	|
|Roundabout mandatory    |   0.65   |   0.78   |   0.71    |    90	|
|End of no passing    |   0.89   |   0.68   |   0.77   |     60	|
|End of no passing by vehicles over 3.5 metric tons    |   0.91   |   0.86   |   0.88   |     90	|
|**avg / total**   |    **0.93**   |   **0.93**   |   **0.93**  |   **12630**	|

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
	
	I started with the sermanet architecture but wasn't getting good results at the very beggining, so I decided to implement the isdia architecture
	to see if the problem was with the architecture implementation or it was something else.
* What were some problems with the initial architecture?
	
	After some testing I realized the architecture was fine since the beggining. Poor results were caused by a wrong hyperparameter choice.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
	
	I designed the architecture inside a function that allowed me to include batch_normalization by changing one of the arguments. After that I kept some hyperparameter values
	and tried using grayscale, rgb and color images, as well as using a different number of feature maps for each convolutional layer.
	After some trials I decided to keep the architecture that performed best overall.

* Which parameters were tuned? How were they adjusted and why?
	
	The most critical hyperparameter to tune was the learning rate, I tried several different magnitudes until the network learned instead of stalling (due to overshooting).
	I started with 100 epochs, but noticed that the network was still learning quite fast, so I decided to increment it to 200, where it seems to plateau.
	Batch size wasn't very relevant, but best values were 256 and 512, being 512 the best.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
	
	Using batch normalization instead of dropout increased the training speed. I usually favor batch normalization over dropout, since they both solve the same kind of problem.
	Batch normalization optimizes the performance of the algorithm, and by design it also indirectly performs regularization since the stats are calculated per mini-batch,
	meaning that the summay statistics might differ from the actual ones from the whole dataset, which in turn creates some noise in the normalized features.

	Something that would help increase the performance of the current network would be to use a [spacial transformer network](https://arxiv.org/pdf/1506.02025.pdf)

If a well known architecture was chosen:
* What architecture was chosen?
	
	Sermanet, as proposed in the paper stated above. It was designed to perform best on traffic signs. But the architecture resembles [RESNet](https://arxiv.org/pdf/1512.03385v1.pdf)
	since it uses skip connections from previous layers to be fed into later ones.
* Why did you believe it would be relevant to the traffic sign application?
	
	It was designed to perform best on traffic signs.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 	
	Although I couldn't reach the same accuracy shown in the paper, I feel that the network is doing a good job at classifying images.
	I think the best way to improve performance would be to improve the preprocessing pipeline as well as adding a spacial transformer network.
	After those changes, increasing epochs to 300 would probably make the network achieve better results.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are nine German traffic signs that I found on the web:

![alt text][image10] ![alt text][image11] ![alt text][image12] ![alt text][image13] 
![alt text][image14] ![alt text][image15] ![alt text][image16] ![alt text][image17]
![alt text][image18]

Some images might be difficult to classify becuase the size is much bigger and resizing will hurt the quality of the final image. Also some of the images are not
centered on the traffic sign and include artifacts that might confuse the network.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)      		| Speed limit (30km/h)  									| 
| Stop     			| Stop										|
| Beware of ice/snow					| Slippery road											|
| Right-of-way at the next intersection	      		| Right-of-way at the next intersection					 				|
| Yield			| Yield      							|
| Children crossing			| Road work      							|
| Keep right			| Beware of ice/snow     							|
| Speed limit (60km/h)			| Speed limit (50km/h)      							|
| Road work			| Road work      							|


The model was able to correctly guess 5 of the 9 traffic signs, which gives an accuracy of 55%. This compares unfavorably to the accuracy on the test set.
The main reason is that the images have an original size much bigger than 32x32 and some of them are not centered on the traffic sign.
When resizing to 32x32 we lose a lot of information and quality on the image, which I assume is the origin of the low accuracy.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For simplicity I will only show 3 of the new images predictions. For the complete predictions refer to the appropriate cell in the notebook.

For the first image, the model is completely sure that this is a Speed limit (30km/h) (probability of 0.955), and it indeed is. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .955         			| Speed limit (30km/h)   									| 
| .012     				| Speed limit (20km/h) 										|
| .009					| Speed limit (50km/h)											|
| .004	      			| Speed limit (60km/h)					 				|
| .003				    | Speed limit (100km/h)      							|

For the second image, the model has trouble figuring out the sign, it is identified as a Slippery road sign (probability of 0.318), 
but it is in fact a Right-of-way at the next intersection sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .318         			| Slippery road   									| 
| .279     				| Bicycles crossing 										|
| .104					| Speed limit (120km/h)											|
| .092	      			| Road work					 				|
| .050				    | Right-of-way at the next intersection      							|

For the third image, the model strongly thinks that this is a Right-of-way at the next intersection sign (probability of 0.884), but it is in fact a Yield sign.
The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .884         			| Right-of-way at the next intersection   									| 
| .101     				| Children crossing 										|
| .008					| Slippery road											|
| .002	      			| Beware of ice/snow					 				|
| .001				    | Bicycles crossing      							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![alt text][image19]

These feature maps beling to the output of the first convolutional layer. As we can see, most feature maps can recognize some kind of edges that define the triangular
shape of the sign. Some of them are blurry and only show the triangle shape, those are probably color sensitive feature maps.
