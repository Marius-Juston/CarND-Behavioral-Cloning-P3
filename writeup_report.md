# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./project_report/download.svg "Model Visualization"
[image2]: ./data/IMG/center_2020_06_03_15_24_45_821.jpg "Center lane image"
[image3]: ./data/IMG/center_2020_06_03_15_25_46_401.jpg "Recovery Image"
[image4]: ./data/IMG/center_2020_06_03_15_25_46_741.jpg "Recovery Image"
[image5]: ./data/IMG/center_2020_06_03_15_25_47_110.jpg "Recovery Image"
[image6]: ./data/IMG/center_2020_06_03_15_25_47_032.jpg "Normal Image"
[image7]: ./project_report/flipped.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network (model.py lines 18-36), that uses 3x3 filter sizes and 5x5 with a stride of 2 kernels, with depths between 24-64. And a linear part that ranged from 10-1164.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 20). EAch of the relu layers have their weights initialized using He uniform algorithm as I read through [here](https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79), that it was the best way to initialize Relu.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 29-35). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 92). The validation and training set were distributed as 20% in validation and the rest as training. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25). I also tried multiple steering angle correction factors from 0.15, 0.2, 0.25, and 0.3, in the end 0.25 seemed to perform the best to bring the car back in track. Another parameter I had to tune was the number of epochs, I tried many epoch and 5 seemed like it was working best. I also used a dropout of 0.5 and that seemed to work well. 

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to first start simple and then use the NVIDIA model to train my dataset on.

My first step was to use a convolution neural network model similar to the one in the lesson I thought this model might be appropriate because it was working well when viewed in the lesson; however, when I tried it the car was not driving correctly or nearly as smoothly as the one shown.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a very low mean squared error on the validation set. This implied that the model images where too similar from the validation and the training set and maybe also some overfitting. 

To combat the overfitting, I modified the model so that it incorporated dropout layers.

Then I moved on to use the NVIDIA CNN model to try and get a better model, I also reduced the cropping region in order for the model to see more of what was happening. I also added dropout layers to the linear part of the model. And I added Batch Normalization to try to make the model more stable.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track especially the area after the bridge where there was a dirt spot, the issue was due to the bacth normalizaation that was normalizing the extreme steeering angle required thus causing the car to not turn enough and thus just go straight to improve the driving behavior in these cases, I removed the batch normalization layers

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. Though it does still swerve a lot in some areas.

#### 2. Final Model Architecture

The final model architecture (model.py lines 19-36) consisted of a convolution neural network with the following layers and layer sizes ...

| Layer (type)              | Output Shape           | Param #   |
|:-------------------------:|:----------------------:|:----------|
| lambda (Lambda)           | (None, 160, 320, 3)    | 0         |
| cropping2d (Cropping2D)   | (None, 75, 320, 3)     | 0         |
| conv2d (Conv2D)           | (None, 36, 158, 24)    | 1824      |
| conv2d_1 (Conv2D)         | (None, 16, 77, 36)     | 21636     |
| conv2d_2 (Conv2D)         | (None, 6, 37, 48)      | 43248     |
| conv2d_3 (Conv2D)         | (None, 4, 35, 64)      | 27712     |
| conv2d_4 (Conv2D)         | (None, 2, 33, 64)      | 36928     |
| flatten (Flatten)         | (None, 4224)           | 0         |
| dense (Dense)             | (None, 1164)           | 4917900   |
| dropout (Dropout)         | (None, 1164)           | 0         |
| dense_1 (Dense)           | (None, 100)            | 116500    |
| dropout_1 (Dropout)       | (None, 100)            | 0         |
| dense_2 (Dense)           | (None, 50)             | 5050      |
| dropout_2 (Dropout)       | (None, 50)             | 0         |
| dense_3 (Dense)           | (None, 10)             | 510       |
| dropout_3 (Dropout)       | (None, 10)             | 0         |
| dense_4 (Dense)           | (None, 1)              | 11        |

Total params: 5,171,319

Trainable params: 5,171,319

Non-trainable params: 0
_________________________________________________________________

Here is a visualization of the architecture:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to return back in track These images show what a recovery looks like starting from the left:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track one but in reverse in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would balance the right and left steering for the model so that it would not just think to steer to the left, thus generalizing the model more. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had 41736 number of data points. I then preprocessed this data by changing the images from RGB to YUV so that the model could better see the curves and also I removed the dataset points that had speeds of less than 25mph so that it would not could the ones where I was just starting to launch.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the fact that the training and validation loss seemed to converge at about that number and were not really changing. I used an adam optimizer so that manually training the learning rate wasn't necessary.
