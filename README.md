# Video Classification
Video classification inspired on [TensorFlow C++ and Python Image Recognition Demo](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/label_image).

This example shows how you can load a pre-trained TensorFlow network and use it to recognize objects in videos.

## Description
`label_video.py` sticks more to the code found on the Demo linked above, it uses the Tensorflow 1.x and Graphs.

`label_video_keras.py` is implemented with keras pretrained models, the script lets you pick between *MobileNet*, *VGG16*, *InceptionV3* and *ResNet50*.

## Usage

### label_video.py
```
usage: label_video.py [-h] [--video VIDEO] [--graph GRAPH] [--labels LABELS]
                      [--input_height INPUT_HEIGHT]
                      [--input_width INPUT_WIDTH] [--input_mean INPUT_MEAN]
                      [--input_std INPUT_STD] [--input_layer INPUT_LAYER]
                      [--output_layer OUTPUT_LAYER]

optional arguments:
  -h, --help            show this help message and exit
  --video VIDEO         filename of the video to be processed
  --graph GRAPH         graph/model to be executed
  --labels LABELS       name of file containing labels
  --input_height INPUT_HEIGHT
                        input height
  --input_width INPUT_WIDTH
                        input width
  --input_mean INPUT_MEAN
                        input mean
  --input_std INPUT_STD
                        input std
  --input_layer INPUT_LAYER
                        name of input layer
  --output_layer OUTPUT_LAYER
                        name of output layer
```

### label_video_keras.py
```
usage: label_video_keras.py [-h] [--video VIDEO] [--model MODEL]

optional arguments:
  -h, --help     show this help message and exit
  --video VIDEO  filename of the video to be processed
  --model MODEL  model to be used, options: 'MobileNet', 'VGG16', 'InceptionV3', 'ResNet50'
```

