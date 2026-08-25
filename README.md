# cascad-ai
The goal of this project is to develop a system that automatically determines the score of a single player's Cascadia game board from a photograph of the completed environment.

## Creating a Dataset

I want to automaically create and annotate a dataset to train a YOLO model.

[How to Create a Dataset for a YOLO model](https://docs.ultralytics.com/yolov5/tutorials/train-custom-data#2-select-a-model)

## Object Detection Strategies
- DNN (YOLO)
- Circle Detection 
    - [Hough Circle Transform](https://docs.opencv.org/3.4.20/d4/d70/tutorial_hough_circle.html)    
- KMeans with color and Distance

## Token Detector Model

Trained on 2000 Artifical images made with the dataset generator tool. Model was trained within 5 epochs. Trained with imgsz param set to 800,

Best model is saved at: `src/cascadai/models/token_detector_yolo11n/TokenNet.pt`

The model has the yolo11n architecture.

Notes on Perfomance:
- really good performance for image size = 1024 and confidence threshold > 0.7

## Ideas 
- remove background to prevent token incorrect token detection
- use hough circle transformation to find locations of tokens
- use a classifier (Auto encoder?) to classify regions where tokens were found 
- use detected tokens to build a graph
- use the detected tokens to sample local hexagons
- train a classifier to classify a hexagon based on local sample region, which is occluded/coverd by a token / or not.