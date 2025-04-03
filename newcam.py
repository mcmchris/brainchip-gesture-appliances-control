#!/usr/bin/env python
"""
Pi Camera Live Image Classification

Detects objects in continuous stream of images from Pi Camera. Use Edge Impulse
Runner and downloaded .eim model file to perform inference. Bounding box info is
drawn on top of detected objects along with framerate (FPS) in top-left corner.

Author: EdgeImpulse, Inc.
Date: August 3, 2021
Updated: August 9, 2023
License: Apache-2.0 (apache.org/licenses/LICENSE-2.0)
"""

import os, sys, time
import cv2
from picamera2 import Picamera2
import requests
import json
import numpy as np
from dotenv import load_dotenv
from flask import Flask, render_template, Response
from edge_impulse_linux.image import ImageImpulseRunner

load_dotenv()

app = Flask(__name__, static_folder='templates/assets')

# Settings
model_file = "model.eim"             # Trained ML model from Edge Impulse
res_width = 192                          # Resolution of camera (width)
res_height = 192                         # Resolution of camera (height)
rotation = 180                            # Camera rotation (0, 90, 180, or 270)
cam_format = "RGB888"                   # Color format

url = 'https://homeassistant.mcmchris.com/api/services/google_assistant_sdk/send_text_command'
auth = os.getenv("HOME_ASSISTANT_KEY")

headers = {    
    "Content-Type": "application/json",
    "authorization": auth
}

# The ImpulseRunner module will attempt to load files relative to its location,
# so we make it load files relative to this program instead
dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(dir_path, model_file)

# Load the model file
runner = ImageImpulseRunner(model_path)

def gen_frames():
    # Initialize model (and print information if it loads)
    try:
        model_info = runner.init()
        print("Model name:", model_info['project']['name'])
        print("Model owner:", model_info['project']['owner'])
        
    # Exit if we cannot initialize the model
    except Exception as e:
        print("ERROR: Could not initialize model")
        print("Exception:", e)
        if (runner):
            runner.stop()
        sys.exit(1)

    # Initial framerate value
    fps = 0

    ACcount = 0
    TVcount = 0
    LIGHTcount = 0 
    OTHERcount = 0

    lightStat = 0
    acStat = 0
    tvStat = 0

    trustVal = 3

    rptCtrl = 0

    # Interface with camera
    with Picamera2() as camera:

        # Configure camera settings
        config = camera.create_video_configuration(
            main={"size": (res_width, res_height), "format": cam_format})
        camera.configure(config)

        # Start camera
        camera.start()
        
        # Continuously capture frames
        while True:
                                                
            # Get timestamp for calculating actual framerate
            timestamp = cv2.getTickCount()
            
            # Get array that represents the image (in RGB format)
            img = camera.capture_array()

            # Rotate image
            if rotation == 0:
                pass
            elif rotation == 90:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                img = cv2.rotate(img, cv2.ROTATE_180)
            elif rotation == 270:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                print("ERROR: rotation not supported. Must be 0, 90, 180, or 270.")
                break
            
            # Extract features (e.g. grayscale image as a 2D array)
            features, cropped = runner.get_features_from_image(img)
            
            # Perform inference
            res = None
            try:
                res = runner.classify(features)
            except Exception as e:
                print("ERROR: Could not perform inference")
                print("Exception:", e)
                
            # Display predictions and timing data
            print("-----")
            results = res['result']['classification']
            for label in results:
                prob = results[label]
                print(label + ": " + str(round(prob, 3)))

                if label == "light" and prob > 0.9:
                    LIGHTcount = LIGHTcount + 1 
                    if LIGHTcount > trustVal and rptCtrl == 1:
                        rptCtrl = 0
                        print("You are pointing the Lightbulb")
                        lightStat = not(lightStat)
                        if lightStat == 1:
                            x = requests.post(url, data=json.dumps({"command":"prende la luz de la habitacion"}), headers=headers)
                        elif lightStat == 0:
                            x = requests.post(url, data=json.dumps({"command":"apaga la luz de la habitacion"}), headers=headers)
                        if x.status_code == 200:
                            print('Lightbulb controlled successfully')
                        LIGHTcount = 0
                if label == "ac" and prob > 0.9:
                    ACcount = ACcount + 1
                    if ACcount > trustVal and rptCtrl == 1:
                        rptCtrl = 0
                        print("You are pointing the Air Conditioner")
                        acStat = not(acStat)
                        if acStat == 1:
                            x = requests.post(url, data=json.dumps({"command":"prende el aire de la habitacion"}), headers=headers)
                        elif acStat == 0:
                            x = requests.post(url, data=json.dumps({"command":"apaga el aire de la habitacion"}), headers=headers)
                        if x.status_code == 200:
                            print('AC controlled successfully')
                        ACcount = 0
                if label == "tv" and prob > 0.9:
                    TVcount = TVcount + 1
                    if TVcount > trustVal and rptCtrl == 1:
                        rptCtrl = 0
                        print("You are pointing the TV")
                        tvStat = not(tvStat)
                        if tvStat == 1:
                            x = requests.post(url, data=json.dumps({"command":"prende la television"}), headers=headers)
                        elif tvStat == 0:
                            x = requests.post(url, data=json.dumps({"command":"apaga la television"}), headers=headers)
                        if x.status_code == 200:
                            print('TV controlled successfully')
                        TVcount = 0
                if label == "other" and prob > 0.9:
                    OTHERcount = OTHERcount + 1
                    if OTHERcount > 2:
                        rptCtrl = 1
                        LIGHTcount = 0
                        ACcount = 0
                        TVcount = 0
                        OTHERcount = 0

            print("FPS: " + str(round(fps, 3)))
            
            # Find label with the highest probability
            max_label = max(results, key=results.get)
            print("Inference: " + max_label)

            # For viewing, convert image to grayscale
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            res, buffer = cv2.imencode('.jpg', img)

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            # Calculate framerate
            frame_time = (cv2.getTickCount() - timestamp) / cv2.getTickFrequency()
            fps = 1 / frame_time
            
            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break        


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)

# Clean up
#cv2.destroyAllWindows()

