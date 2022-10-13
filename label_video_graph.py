# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import numpy as np
import tensorflow.compat.v1 as tf
import cv2
import time

np.set_printoptions(suppress=True)


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_frame(sess,
                           frame,
                           input_height=299,
                           input_width=299,
                           input_mean=0,
                           input_std=127):
    float_caster = tf.cast(frame, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    return sess.run(normalized)


def load_labels(label_file):
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    return [l.rstrip() for l in proto_as_ascii_lines]


if __name__ == "__main__":
    file_name = 'data/ladybug.mp4'
    model_file = 'data/inception_v3_2016_08_28_frozen.pb'
    label_file = "data/imagenet_slim_labels.txt"
    input_layer = "input"
    output_layer = "InceptionV3/Predictions/Reshape_1"
    input_mean = 0
    input_std = 127
    # InceptionV3 Input height and width
    input_height = 299
    input_width = 299

    tf.compat.v1.disable_eager_execution()
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="video to be processed")
    parser.add_argument("--graph", help="graph/model to be executed")
    parser.add_argument("--labels", help="name of file containing labels")
    parser.add_argument("--input_height", type=int, help="input height")
    parser.add_argument("--input_width", type=int, help="input width")
    parser.add_argument("--input_mean", type=int, help="input mean")
    parser.add_argument("--input_std", type=int, help="input std")
    parser.add_argument("--input_layer", help="name of input layer")
    parser.add_argument("--output_layer", help="name of output layer")
    args = parser.parse_args()

    if args.graph:
        model_file = args.graph
    if args.video:
        file_name = args.video
    if args.labels:
        label_file = args.labels
    if args.input_height:
        input_height = args.input_height
    if args.input_width:
        input_width = args.input_width
    if args.input_mean:
        input_mean = args.input_mean
    if args.input_std:
        input_std = args.input_std
    if args.input_layer:
        input_layer = args.input_layer
    if args.output_layer:
        output_layer = args.output_layer

    graph = load_graph(model_file)

    cap = cv2.VideoCapture(file_name)

    if not(cap.isOpened()):
        print("File not found: {}".format(file_name))

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)
    labels = load_labels(label_file)


    with tf.compat.v1.Session(graph=graph) as sess:
        while cap.isOpened():
            ret, img = cap.read()

            # If there are no frames left, exit while loop
            if not(ret):
                break

            # When the frame is read the order of colors is BGR (blue, green, red)
            # Next line converts it to RGB
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
            t = read_tensor_from_frame(
                sess,
                imgRGB,
                input_height=input_height,
                input_width=input_width,
                input_mean=input_mean,
                input_std=input_std)
             
            start = time.time()
            results = sess.run(output_operation.outputs[0], {
                input_operation.outputs[0]: t
            })
            end = time.time()

            results = np.squeeze(results)
            index = np.argmax(results)

            class_name = labels[index]
            confidence_score = results[index]
            confidence_level = confidence_score * 100

            totalTime = end - start
            fps = 1 / totalTime

            print("Class: ", class_name)
            print("Confidence score: ", confidence_score)
            print("FPS: {:.2f}".format(fps))

            if confidence_level > 80:
                text_color = (0, 255, 0)
            else:
                text_color = (0, 0, 255)

            cv2.putText(img, str(float("{:.2f}".format(confidence_level))) + "% " +
                        class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        text_color, 2)

            cv2.imshow('Video Classification', img)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    cap.release()
