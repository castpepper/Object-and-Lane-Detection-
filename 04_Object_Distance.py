import numpy as np
import os
import sys
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import cv2

# This line is the only difference between thw Camera and Video files being executed in the Tensorflow program.
cap = cv2.VideoCapture("Object_Video.mp4")
# cap = cv2.VideoCapture(0)
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# # Model preparation

# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT`
# to point to a new .pb file.
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo]
# (https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md)
# for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# What model to use.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# ## Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`,
# we know that this corresponds to `airplane`.  Here we use internal utility functions,
# but anything that returns a dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


# # Detection

PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3)]

# Size, in inches, of the output images.

IMAGE_SIZE = (12, 8)

# Final Output Module which works on the input to get the segregated objects in the output.

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        while True:
            ret, image_np = cap.read()
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(image_np, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index, use_normalized_coordinates=True, line_thickness=8)

            for i, b in enumerate(boxes[0]):
                if classes[0][i] == 3 or classes[0][i] == 6 or classes[0][i] == 8:
                    if scores[0][i] > 0.4:
                        mid_x = (boxes[0][i][3] + boxes[0][i][1]) / 2
                        mid_y = (boxes[0][i][2] + boxes[0][i][0]) / 2
                        apx_distance = round((1 - (boxes[0][i][3] - boxes[0][i][1]))**4, 1)
                        cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800), int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        if apx_distance <= 0.5:
                            if mid_x > 0.3 and mid_x < 0.7:
                                cv2.putText(image_np, 'WARNING!!!', (int(mid_x*800)-50, int(mid_x*450)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            cv2.namedWindow('cam', cv2.WINDOW_NORMAL)
            cv2.imshow('cam', image_np)
            key = cv2.waitKey(1) & 0xFF
            # if the q key was pressed, break from the loop
            if key == ord("A"):
                break

cap.release()
cv2.destroyAllWindows()
