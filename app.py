import time
from absl import app, logging
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
from flask import Flask, request, Response, jsonify, send_from_directory, abort
import os


CLASSES_PATH = './data/labels/coco.names'
UNALLOWED_CLASSES_PATH = './data/labels/unallowed.names'
WEIGHTS_PATH = './weights/yolov3.tf'
TINY = False                    # set to True if using a Yolov3 tiny model
SIZE = 512 #416                      # Size images are resized to for model
OUTPUT_PATH = './detections/'   # path to output folder where images with detections are saved             
NUM_CLASSES = 80 # number of classes in model

class_names = [c.strip() for c in open(CLASSES_PATH).readlines()]
print('classes loaded')
unallowed_class_names = set([c.strip() for c in open(UNALLOWED_CLASSES_PATH).readlines()])
print('unallowed classes loaded')

# load in weights and classes
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

if TINY:
    yolo = YoloV3Tiny(classes=NUM_CLASSES)
else:
    yolo = YoloV3(classes=NUM_CLASSES)

yolo.load_weights(WEIGHTS_PATH).expect_partial()
print('weights loaded')


# Initialize Flask application
app = Flask(__name__)

# Returns JSON with classes found in images
@app.route('/detections', methods=['POST'])
def get_detections():
    try:
        raw_images = []
        images = request.files.getlist("images")
        image_names = []
        for image in images:
            image_name = image.filename
            image_names.append(image_name)
            image.save(os.path.join(os.getcwd(), image_name))
            img_raw = tf.image.decode_image(
                open(image_name, 'rb').read(), channels=3)
            raw_images.append(img_raw)
            
        num = 0
        
        # create list for final response
        response = []

        for j in range(len(raw_images)):
            # create list of responses for current image
            objects = []
            raw_img = raw_images[j]
            num+=1
            img = tf.expand_dims(raw_img, 0)
            original_size = img.shape[1:3]
            img = transform_images(img, SIZE)

            t1 = time.time()
            boxes, scores, classes, nums = yolo(img)
            t2 = time.time()
            print('time: {}'.format(t2 - t1))

            # print('detections:')
            for i in range(nums[0]):
                '''
                print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                                np.array(scores[0][i]),
                                                np.array(boxes[0][i])))
                '''
                if class_names[int(classes[0][i])] in unallowed_class_names:
                    x1y1 = ((np.array(boxes[0][i][0:2]) * original_size).astype(np.int32)).tolist()
                    wh = ((np.array(boxes[0][i][2:4]) * original_size).astype(np.int32)).tolist() - x1y1
                    objects.append({
                        "label": class_names[int(classes[0][i])],
                        "confidence": float("{0:.2f}".format(np.array(scores[0][i])*100)),
                        "x": x1y1[0],
                        "y": x1y1[1],
                        "w": wh[0],
                        "h": wh[1]
                    })
            response.append({
                "image": image_names[j],
                "detections": objects
            })
            save_image(raw_img, num, (boxes, scores, classes, nums))
        #remove temporary images
        for name in image_names:
            os.remove(name)
        
        try:
            return jsonify(response), 200
        except FileNotFoundError:
            abort(404)
        
    except tf.errors.InvalidArgumentError as e:
        return jsonify({'message':'Wrong file type used'}), 400
    except FileNotFoundError as e:
        return jsonify({'message':e.args}), 404

# Returns image with detections on it
@app.route('/image', methods= ['POST'])
def get_image():
    try:
        image = request.files["images"]
        image_name = image.filename
        image.save(os.path.join(os.getcwd(), image_name))
        img_raw = tf.image.decode_image(
            open(image_name, 'rb').read(), channels=3)
        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, SIZE)

        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        t2 = time.time()
        
        print('time: {}'.format(t2 - t1))
        '''
        print('detections:')
        for i in range(nums[0]):
            print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                            np.array(scores[0][i]),
                                            np.array(boxes[0][i])))
        '''
        img = save_image(img_raw, 0, (boxes, scores, classes, nums), return_img=True)
        # prepare image for response
        _, img_encoded = cv2.imencode('.png', img)
        response = img_encoded.tostring()
        
        #remove temporary image
        os.remove(image_name)

        try:
            return Response(response=response, status=200, mimetype='image/png')
        except FileNotFoundError:
            abort(404)
    except tf.errors.InvalidArgumentError as e:
        return jsonify({'message':'Wrong file type used'}), 400
    except FileNotFoundError as e:
        return jsonify({'message':e.args}), 404

def save_image(raw_img, num, outputs, return_img=False):
    img = cv2.cvtColor(raw_img.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, outputs, class_names, unallowed_class_names)
    cv2.imwrite(OUTPUT_PATH + 'detection' + str(num) + '.jpg', img)
    print('output saved to: {}'.format(OUTPUT_PATH + 'detection' + str(num) + '.jpg'))
    if return_img:
        return img
    return None

if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port=5000)
