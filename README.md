
# **Traffic Sign Recognition** 

**Objective :**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarizing the results

[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/RGB_Image.jpg "RGB_Image"
[image5]: ./examples/Gray_Image.jpg "Gray_Image"
[image6]: ./examples/Data_Visualization_1.JPG "Data_Visualization_1"
[image7]: ./examples/Data_Visualization_2.JPG "Data_Visualization_2"


[Test1]: ./Test_Images/3.jpg "1"
[Test2]: ./Test_Images/4.jpg "2"
[Test3]: ./Test_Images/11.jpg "3"
[Test4]: ./Test_Images/12.jpg "4"
[Test5]: ./Test_Images/13.jpg "5"
[Test6]: ./Test_Images/14.jpg "6"
[Test7]: ./Test_Images/25.jpg "7"
[Test8]: ./Test_Images/33.jpg "8"
[Test9]: ./Test_Images/38.jpg "9"

[Result1]: ./examples/Prediction_1.JPG "P_1"
[Result2]: ./examples/Prediction_2.JPG "P_2"
[Result3]: ./examples/Prediction_3.JPG "P_3"
[Result4]: ./examples/Prediction_4.JPG "P_4"
[Result5]: ./examples/Prediction_5.JPG "P_5"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
> *Number of training examples = 34799*
* The size of the validation set is ?
>*Number of testing examples   = 4410*
* The size of test set is ?
>*Number of testing examples  = 12630*
* The shape of a traffic sign image is ?
>*Image data shape            = (32, 32, 3)*
* The number of unique classes/labels in the data set is ?
>*Number of classes           = 43*

#### 2. Include an exploratory visualization of the dataset.

* Visualizing the distribution of data by plotting output classes against the total number of data present in that class 

![alt text][image6]

> From the above plot, It is clear that the output classes are not equally distributed in the training data set.

>> ***Solution*** : By performing `Data Augumentation` (example : rotating imgae, changing brightness), we can achieve equal distribution

* Viewing the image and understanding the relationship with the output class

![alt text][image7]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Since the images are taken in different environment, We are `Pre-process` our data set before feeding into neural network. In this project, The below two pre-processing technique is followed. 
> 1. Normalization
>>By performing normalization, we can achieve zero mean and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 
> 2. RGB to Gray
>> By reducting the color channel, we can reduce the computational cost and increase the accuracy in detecting the shape rather than color.
>
> RGB Image
>> ![alt text][image4]
>
> Gray Image
>> ![alt text][image5]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I took LeNet architecture as base and implemented the Traffic sign classifier

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| ReLU Activation														|
| Max pooling	      	| 2x2 stride,  2x2 filter, outputs 14x14x6		|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| ReLU Activation														|
| Max pooling	      	| 2x2 stride,  2x2 filter, outputs 5x5x16		|
| Fully connected#1		| 400x120 input, 120x84 output					|
| ReLU Activation														|
| Drop out				| Drop out percentage 65%						|
| Fully connected#2		| 120x84 input, 84x43 output					|
| ReLU Activation														|
| Drop out				| Drop out percentage 85%						|
| Fully connected#3		| 120x84 input, 84x43 output					|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

**Model 1:** Epochs: 20, Learning rate: 0.001 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| SoftMax Activation													|
| Average pooling	    | 2x2 stride,  2x2 filter, outputs 14x14x6	    |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| SoftMax Activation													|
| Average pooling	    | 2x2 stride,  2x2 filter, outputs 5x5x16	    |
| Fully connected#1		| 400x120 input, 120x84 output					|
| SoftMax Activation													|
| Fully connected#2		| 120x84 input, 84x43 output					|
| SoftMax Activation													|
| Fully connected#3		| 120x84 input, 84x43 output					|

In this model, The validation accuracy is raise upto in ~9%. The accuracy level is too low. We can increase accuray with increase in epochs. But, I belive that before tuning our epochs size, we can change the model. As first step, I have changed the following
> 1. Activation function as "ReLU"
> 2. Changed the pooling method from Average to Max

**Model 2:** Epochs: 20, Learning rate: 0.001 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| ReLU Activation														|
| Max pooling	      	| 2x2 stride,  2x2 filter, outputs 14x14x6		|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| ReLU Activation														|
| Max pooling	      	| 2x2 stride,  2x2 filter, outputs 5x5x16		|
| Fully connected#1		| 400x120 input, 120x84 output					|
| ReLU Activation														|
| Fully connected#2		| 120x84 input, 84x43 output					|
| ReLU Activation														|
| Fully connected#3		| 120x84 input, 84x43 output					|

In this model, The validation accuracy is raised upto ~70. To further increase the accuracy, Drop out has been introduced after the fully connected layer.


**Model 3:** Epochs: 20, Learning rate: 0.001 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| ReLU Activation														|
| Max pooling	      	| 2x2 stride,  2x2 filter, outputs 14x14x6		|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| ReLU Activation														|
| Max pooling	      	| 2x2 stride,  2x2 filter, outputs 5x5x16		|
| Fully connected#1		| 400x120 input, 120x84 output					|
| ReLU Activation														|
| Drop out				| Drop out percentage 65%						|
| Fully connected#2		| 120x84 input, 84x43 output					|
| ReLU Activation														|
| Drop out				| Drop out percentage 85%						|
| Fully connected#3		| 120x84 input, 84x43 output					|

With this model, I got around 89%. By tuning the hyper parameters, we can acheive the desired output. So, this model is selected for training.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

**Type 1 :** Epochs: 20, Learning rate: 0.001 
The validation accuracy is starts in ~65% and goes upto ~90%. To achieve the good accuracy, we can perform the following ***TWO ways***. They are,
> 1. Increase the Epochs 
> 2. Incraese the Learning Rate

**Type 2:** Epochs: 50, Learing rate: 0.001
The validation accuracy is saturated on ~93 after 20 Epochs. The accuracy is fluctuating after 30 Epochs which may due to overfitting. By reducing the epchos and increasing the learning rate, we can further improve the accuracy.


**Type 3:** Epochs: 20, Learing rate: 0.004
After tring different combination of epochs & learning rate, The following combination best fits my train data.
> 1. Epochs        : 20 
> 2. learning rate : 0.004

My final model results were:
* training set accuracy of ?
>* `Train Accuracy = 99.569`
* validation set accuracy of ? 
>* `Valid Accuracy = 95.215`
* test set accuracy of ?
>* `Test Accuracy = 93.658`

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
> I started with LeNet Architecture. Because I have work experience in that and `LeNet architecture` worked well on MNIST data set. Hence I adapted the same to classify the traffic sign images.

* What were some problems with the initial architecture?
> I found difficulty in finding the activation function and pooling methods. By experimenting different combination, I solved this issue.

* How was the architecture adjusted and why was it adjusted? 
> There were 2 major adjustment done while building the model.
> 1. Finding the optimal activation function - To improve the `train_accuracy`
> 2. Introducing drop out function after the fully connected layer - To make neural network less depend on train data. In turn, this will increase the `Validation_accuracy`

* Which parameters were tuned? How were they adjusted and why?
> I have changed the learning rate and epochs. They are inversely propotional to each other. I chose large the learning rate with small epochs.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
>1. **Convolution layer** : This uses the weight sharing which helps in improving translation invariance.
>2. **Drop Out** : This will make our neural network depends on the features of the data rather than the pixel value.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are Nine German traffic signs that I found on the web:

![alt text][Test1] 
![alt text][Test2] 
![alt text][Test3] 
![alt text][Test4] 
![alt text][Test5] 
![alt text][Test6] 
![alt text][Test7] 
![alt text][Test8] 
![alt text][Test9] 


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection| Right-of-way at the next intersection| 
| Priority road     	| Ahead only									|
| Yield					| Yield											|
| Stop	      			| Priority road					 				|
| Road work				| Ahead only	      							|
| Speed limit (60km/h) 	| Speed limit (60km/h)							|
| Turn right ahead		| Speed limit (30km/h)							|
| Keep right			| Keep right									|
| Speed limit (70km/h) 	| Speed limit (30km/h)							|


The model was able to correctly guess  4 of the 9 traffic signs, which gives an accuracy of 44.44%.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 21th cell of the Ipython notebook.

The following image will depicts the top 5 prediction of each input.
![alt text][Result1]
![alt text][Result2]
![alt text][Result3]
![alt text][Result4]
![alt text][Result5]

### Analysis of output
The following `Five` images are not classified correctly
> 1. Priority Road
> 2. Road Work
> 3. Stop
> 4. Turn Right Ahead
> 5. Speed Limit(70Km/h)

##### Major Reasons for failure:
1. Our training data set has traffic sign at the center part of the image where as our chosen set has traffic sign in different part of the image.
2. Image quality is not good in some photos.(Road Work)
3. Input images are rotated(i.e., the photos taken in different angle)
4. Brightness level of image set is varied
5. Data is not uniformly spread across the training data set(i.e., if the count of the a particular label/class is small compare to other label in our training data, then there is possibility of mis-classification of that label)

