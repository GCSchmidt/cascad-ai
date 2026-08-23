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

Trained on 1000 Artifical images made with the dataset generator tool. Model was trained within 4 epochs.

Best model is saved at: `src/cascadai/models/token_detector_yolo11n/TokenNet.pt`

The model has the yolo11n architecture.

Notes on Perfomance:
- model struggles the most with deer tokens
- play around with confidence threshold and image size
- sometime deers are detected when hexagons are missing on wooden surface.

## Ideas 
- remove background to prevent token incorrect token detection
- use hough circle transformation to find locations of tokens
- use a classifier (Auto encoder?) to classify regions where tokens were found 