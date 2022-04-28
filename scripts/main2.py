import cv2
import os
#import tensorflow as tf
#import argparse
from flask import Flask, redirect, url_for, render_template, Response
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objectsfrom pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

app = Flask(__name__)

@app.route('/')
def index(): 
    return render_template('index.html')

@app.route('/video_feed')
def camera1_feed(): 
    return Response(main(), mimetype='multipart/x-mixed-replace; boundary=frame')

def main():

    state = 1; #Options: 1, 2, 3, 4
    fps = 10; #Options: 10, 20, 30
    width = 640 #Options: 640, 1280, 1920
    height = 480 #Options: 480, 720, 1080
    recording = False;
    alarm = False;
    threshold = 0.7

    model_dir = '../models'
    model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    model_path = '../models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    label_dir = '../labelmaps'
    labels = 'coco_labels.txt'
    label_path = '../labelmaps/coco_labels.txt'

    model_dir2 = '../models'
    model2 = 'quantized_model.tflite'
    model2_path = '../models/quantized_model.tflite'

    cap1 = cv2.VideoCapture(0)
    cap1.set(cv2.CAP_PROP_FPS, fps)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    #cap2 = cv2.VideoCapture(1)
    #cap2.set(cv2.CAP_PROP_FPS, fps)
    #cap2.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    #cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    #cap3 = cv2.VideoCapture(2)
    #cap3.set(cv2.CAP_PROP_FPS, fps)
    #cap3.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    #cap3.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    #cap4 = cv2.VideoCapture(3)
    #cap4.set(cv2.CAP_PROP_FPS, fps)
    #cap4.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    #cap4.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    while (True):

        ret, frame = cap1.read()
        if not ret:
            break
        cv2_im = frame

       #interpreter, inference_size = load_object_detection(model_path, label_path)

        interpreter = make_interpreter(model_path)
        interpreter.allocate_tensors()
        labels = read_label_file(label_path)
        inference_size = input_size(interpreter)


        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        run_inference(interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(interpreter, threshold)

        interpreter = make_interpreter(model2_path)
        interpreter.allocate_tensors()
        inference_size = input_size(interpreter)

        run_inference(interpreter, cv2_im_rgb.tobytes())

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        cv2_im = append_objs_to_img(cv2_im, inference_size, objs, labels)

        cv2.imshow('frame', cv2_im)

        if (state == 1):
            for obj in objs:
                if (obj.id == 0 or obj.id == 1 or obj.id == 2 or obj.id == 3):
                    #Target detected, transition to state 2
                    state = 2
                    cap1.set(cv2.CAP_PROP_FPS, fps)
                    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                else:
                    print("no")
                    #No target detected, stay in state 1

        elif (state == 2):
            print("state 2")
            #Target moves closer to warehouse, transition to state 3
            #Target stays far, but moves no closer, stay in state 2
            #Target leaves camera range, transition back to state 1
        elif (state == 3):
            print("state 3")
            #Target continues to move closer, stays close for more than 1 minute, transition to state 4
            #Target continues to move closer, stays close for less than 1 minute, stay in state 3
            #Target moves away, transition back to state 2
        elif (state == 4):
            print("state 4")
            #Target continues to move closer, stays close stay in state 4
            #Target moves away, transition back to state 3` 

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cv2.destroyAllWindows()

def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im


def changeToState1():
    cap.set

def changeToState2():
    print("change")

def changeToState3():
    print("change")

def changeToState4():
    print("change")


if __name__ == '__main__':
    main()
    #app.run(debug=True)
