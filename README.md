# Face Mask Detection using Deep Learning Algorithms

## 1. Overview
Our project aims to detect medical masks in images using a combination of computer vision techniques and deep learning. It utilizes OpenCV for image processing, Pandas and NumPy for data manipulation, and Keras for building and training the deep learning model.

## 2. Prerequisites
Ensure that you have the following dependencies installed:
- Python 3.x
- OpenCV (cv2)
- Pandas
- NumPy
- Keras
- Matplotlib
- Seaborn

Use the following code to install the dependencies:

  `pip install opencv-python pandas numpy keras matplotlib seaborn`

Additionally, you need to add two files related to OpenCV:

- deploy.prototxt.txt: A text file containing the architecture of the neural network used for face detection.

- res10_300x300_ssd_iter_140000.caffemodel: A pre-trained Caffe model for face detection.

Both of these files are available in our GitHub repository

## 3. Usage
- Download the Dataset from this link
    <https://www.kaggle.com/datasets/wobotintelligence/face-mask-detection-dataset?resource=download&select=Medical+mask>
- Download the code from our Repository and run it on any Code Editor using

  `python your_script.py`

  or just run it on Jupyter Notebook or Google Colab

## 4. Output
- The code generates visualizations of mask detection results on test images.
- It also outputs performance metrics such as accuracy during model training.

  ![OutputImage](/Output.png)
