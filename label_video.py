import argparse
import time

from tensorflow.keras.preprocessing import image
import numpy as np
import cv2


if __name__ == "__main__":
    file_name = 'data/ladybug.mp4'

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="video to be processed")
    parser.add_argument("--model", help="model to be used, options: 'MobileNet'"
                                        ", 'VGG16', 'InceptionV3', 'ResNet50'")

    args = parser.parse_args()

    if args.video:
        file_name = args.video

    if args.model:
        model_name = args.model
    else:
        # If model_name not specified, use fastest model
        model_name = 'MobileNet'

    size = (224, 224)  # Model input size for MobileNet, VGG16 and ResNet50

    if model_name == 'MobileNet':
        from tensorflow.keras.applications.mobilenet import MobileNet
        from tensorflow.keras.applications.mobilenet import (preprocess_input,
                                                             decode_predictions)
        model = MobileNet(weights='imagenet')

    elif model_name == 'VGG16':
        from tensorflow.keras.applications.vgg16 import VGG16
        from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                         decode_predictions)
        model = VGG16(weights='imagenet')

    elif model_name == 'ResNet50':
        from tensorflow.keras.applications.resnet50 import ResNet50
        from tensorflow.keras.applications.resnet50 import (preprocess_input,
                                                            decode_predictions)
        model = ResNet50(weights='imagenet')

    elif model_name == 'InceptionV3':
        from tensorflow.keras.applications.inception_v3 import InceptionV3
        from tensorflow.keras.applications.inception_v3 import (preprocess_input,
                                                                decode_predictions)
        size = (299, 299)  # Model input size for InceptionV3
        model = InceptionV3(weights='imagenet')

    else:
        print('Model not found')
        exit()

    cap = cv2.VideoCapture(file_name)
    if not(cap.isOpened()):
        print("File not found: {}".format(file_name))

    while cap.isOpened():
        start = time.time()
        ret, img = cap.read()

        # If there are no frames left, exit loop
        if not(ret):
            break

        # When the frame is read the order of colors is BGR (blue, green, red)
        # Next line converts it to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channels = img.shape
        scale_value = width / height
        img_resized = cv2.resize(imgRGB, size, fx=scale_value, fy=1,
                                 interpolation=cv2.INTER_NEAREST)

        x = image.img_to_array(img_resized)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)

        top_prediction = decode_predictions(preds, top=1)[0][0]

        class_name = top_prediction[1]
        confidence_score = top_prediction[2]
        confidence_level = confidence_score * 100

        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime

        print("Model: ", model_name)
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
