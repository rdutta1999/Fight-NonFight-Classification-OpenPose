# Fight-NonFight-Classification-OpenPose

## Introduction
Our aim was to classify Fighting images vs Non-Fighting images, and we wanted to tackle this problem a bit differently. In this repo, we used the Pose Points of the humans in the images and then trained a Support Vector Machine, Logistic Regressor, Neural Network and a VGG16 based CNN to classify the images. To get the pose points, we used [OpenPose](https://arxiv.org/pdf/1812.08008.pdf), a state-of-the-art pose estimation model. However, since OpenPose was originally written in Caffe, we used a [Tensorflow implementation](https://github.com/ildoonet/tf-pose-estimation).

The SVM, Logistic Regressor and Neural Network were completely trained using the Pose Points only as the input, while the VGG16 based CNN used both the Pose Points and the image as the input. The features extracted from the image by the VGG16 segment were then concatenated with the Pose Points, and passed through a number of Dense layers. 

The dataset images, along with the trained models can be found [here](https://drive.google.com/drive/folders/1srtiO6IZhfqyxoaKGmGdCg6YtcKI4Frn?usp=sharing).

## Setting Up
1) Create a new Python/Anaconda environment (optional but recommended). You might use the `environment.yml` file for this purpose.

2) Clone the TF-based [OpenPose](https://github.com/ildoonet/tf-pose-estimation) repo, and set it up.

3) Download the dataset (and the saved models if required) from [here](https://drive.google.com/drive/folders/1srtiO6IZhfqyxoaKGmGdCg6YtcKI4Frn?usp=sharing). Extract and place it in the following manner:-
<pre>
├─── Fight-NonFight-Classification-OpenPose
     ├─── ..
     ├─── docker     
     ├─── etcs     
     ├─── models
     ├─── scripts
     ├─── tf_pose
     ├─── models_info_imgs
     ├─── openpose_output_imgs
     ├─── saved_models
     ├─── ..
     ├─── TrainPosePoints.csv
     ├─── ValidPosePoints.csv     
     ├─── ..
     ├─── environment.yml
     ├─── Run.ipynb
     ├─── Train.ipynb  
     └─── dataset_imgs 
           ├─── fighting
               ├─── ..
               ├─── ..
               └─── .. (200 images) 
           └─── not_fighting
               ├─── ..
               ├─── ..
               └─── .. (200 images) 
</pre>

- `Train.ipynb` --> used to train the five different models.
- `Run.ipynb` --> a demo code to extract frames from a video frame and classify them using the VGG16 based CNN.
- `TrainPosePoints.csv` --> contains the pose points of images in the Training set.
- `ValidPosePoints.csv` --> contains the pose points of images in the Validation set.


## Accuracy and Error Plots
![Pic1](./models_info_imgs/Train-Accuracy.jpg?raw=true) ![Pic2](./models_info_imgs/Valid-Accuracy.jpg?raw=true) ![Pic3](./models_info_imgs/Train-Loss.jpg?raw=true) ![Pic4](./models_info_imgs/Valid-Loss.jpg?raw=true) 
