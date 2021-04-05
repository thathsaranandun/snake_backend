import base64
import io
from io import BytesIO
import json
import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request
from flask_cors import CORS, cross_origin
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util

import DatabaseConnection

CURR_DIR = os.getcwd()
CURR_DIR = CURR_DIR.replace("\\","/")
print(CURR_DIR)
# main_dir = CURR_DIR.split("/snake_backend")[0]
# print(main_dir)
sys.path.append(CURR_DIR+"/models")
sys.path.append(CURR_DIR+"/models/research")
sys.path.append(CURR_DIR+"/models/research/slim")


app = Flask(__name__)
CORS(app, support_credentials=True)
detect_fn = tf.saved_model.load(
    CURR_DIR+"/colab_resnet/saved_model")


@app.route('/status')
def status():
    return 'running...'


@app.route('/api/detect', methods=['POST'])
@cross_origin(supports_credentials=True)
def detect():
    if request.method == 'POST':
        print('[detect] POST request received')
        data = request.get_json(force=True)
        base64_img = data['image']

        try:
            base64_decoded = base64.b64decode(base64_img)

            image = Image.open(io.BytesIO(base64_decoded))
        except:
            print("Invalid image")
            return "Invalid image", 400



        # Changed resolutions
        res = (800, 600)
        im_resized = image.resize(res, Image.ANTIALIAS)

        PATH_TO_LABELS = CURR_DIR+'/resnet/labelmap.pbtxt'
        # MODEL_FILE = 'mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz'

        category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

        image_np = np.array(im_resized)
        image = np.asarray(image_np)
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]
        output_dict = detect_fn(input_tensor)

        num_detections = int(output_dict.pop('num_detections'))
        need_detection_key = ['detection_classes', 'detection_boxes', 'detection_masks', 'detection_scores']
        output_dict = {key: output_dict[key][0, :num_detections].numpy()
                       for key in need_detection_key}
        output_dict['num_detections'] = num_detections
        # detection_classes should be ints.
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
        # Handle models with masks:
        if 'detection_masks' in output_dict:
            # Reframe the the bbox mask to the image size.
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                tf.convert_to_tensor(output_dict['detection_masks']), output_dict['detection_boxes'],
                image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                               tf.uint8)
            output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

        # # Visualization of the results of a detection.

        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None).astype(np.uint8),
            use_normalized_coordinates=True,
            line_thickness=8)
        print(output_dict)

        # convert numpy array to PIL Image
        im = Image.fromarray(image_np)
        buffered = BytesIO()
        im.save(buffered, format="JPEG")

        # move to beginning of file so `send_file()` it will read from start

        print(output_dict['detection_scores'][0])
        print(output_dict['detection_classes'][0])
        print(base64.b64encode(image_np))

        if output_dict['detection_scores'][0] < 0.6:
            return "Oops. No snake detected.", 400

        connection = DatabaseConnection.connectdb()

        try:
            with connection.cursor() as cursor:
                sql = "SELECT * FROM snake_details WHERE id = %s"
                try:
                    cursor.execute(sql, int(output_dict['detection_classes'][0]))
                    result = cursor.fetchall()
                    for row in result:
                        message = {"scientific_name": row['scientific_name'], "sinhala_name": row['sinhala_name'],
                                   "english_name": row['english_name'], "venom": row['venom'], "image": row['image']}
                        return json.dumps({"accuracy": str(output_dict['detection_scores'][0]),
                                           "class": str(output_dict['detection_classes'][0]),
                                           "image":str(base64.b64encode(buffered.getvalue())),
                                           "details":message
                                           })
                except:
                    print("DB Querry Failed!")
                    return "Oops. An error occurred.", 500

        except:
            print('Connection to db failed!')
            return "Oops. An error occurred.", 500


@app.route('/api/snakes')
def get_snakes():
    if request.method == 'GET':
        print('[get_snakes] GET request received')

        connection = DatabaseConnection.connectdb()

        try:
            with connection.cursor() as cursor:
                sql = "SELECT * FROM snake_details"
                try:
                    cursor.execute(sql)
                    result = cursor.fetchall()
                    print(result)
                    return json.dumps({'snakes': result})
                except:
                    print("DB Query Failed!")
                    return "Oops. An error occurred.", 500

        except:
            print('Connection to db failed!')
            return "Oops. An error occurred.", 500


if __name__ == '__main__':
    app.run()
