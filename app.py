import cv2
import os
import time
import numpy as np
import requests
import json
from dotenv import load_dotenv
from flask import Flask, render_template, Response
from edge_impulse_linux.image import ImageImpulseRunner

load_dotenv()

app = Flask(__name__, static_folder='templates/assets')

runner = None
countPeople = 0
inferenceSpeed = 0
videoCaptureDeviceId = int(0) # use 0 for web camera

url = 'https://homeassistant.mcmchris.com/api/services/google_assistant_sdk/send_text_command'
auth = os.getenv("HOME_ASSISTANT_KEY")


headers = {    
    "Content-Type": "application/json",
    "authorization": auth
}

def now():
    return round(time.time() * 1000)

def gen_frames():  # generate frame by frame from camera
    dir_path = os.path.dirname(os.path.realpath(__file__))
    modelfile = os.path.join(dir_path, 'model.eim')
    print('MODEL: ' + modelfile)
    global countPeople
    global inferenceSpeed

    ACcount = 0
    TVcount = 0
    LIGHTcount = 0 
    OTHERcount = 0

    lightStat = 0
    acStat = 0
    tvStat = 0

    trustVal = 3

    rptCtrl = 0

    while True:
        #//////////////////////////////////////////////////////////////////////////////
        with ImageImpulseRunner(modelfile) as runner:
            try:
                model_info = runner.init()
                print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')
                labels = model_info['model_parameters']['labels']
                
                camera = cv2.VideoCapture(videoCaptureDeviceId)
                ret = camera.read()[0]
                
                if ret:
                    backendName = "dummy" #backendName = camera.getBackendName() this is fixed in opencv-python==4.5.2.52
                    w = camera.get(3)
                    h = camera.get(4)
                    print("Camera %s (%s x %s) in port %s selected." %(backendName,h,w, videoCaptureDeviceId))
                    camera.release()
                else:
                    raise Exception("Couldn't initialize selected camera.")
                
                next_frame = 0 # limit to ~10 fps here
                
                for res, img in runner.classifier(videoCaptureDeviceId):
                    count = 0
                    
                    if (next_frame > now()):
                        time.sleep((next_frame - now()) / 1000)

                    # print('classification runner response', res)

                    if "classification" in res["result"].keys():
                        inferenceSpeed =  res['timing']['dsp'] + res['timing']['classification']
                        #print('Result (%d ms.) ' % (res['timing']['dsp'] + res['timing']['classification']), end='')
                        for label in labels:
                            score = res['result']['classification'][label]
                            if label == "light" and score > 0.9:
                                LIGHTcount = LIGHTcount + 1 
                                if LIGHTcount > trustVal and rptCtrl == 1:
                                    rptCtrl = 0
                                    print("You are pointing the Lightbulb")
                                    lightStat = not(lightStat)
                                    if lightStat == 1:
                                        x = requests.post(url, data=json.dumps({"command":"turn on the light"}), headers=headers)
                                    elif lightStat == 0:
                                        x = requests.post(url, data=json.dumps({"command":"turn off the light"}), headers=headers)
                                    if x.status_code == 200:
                                        print('Lightbulb controlled successfully')
                                    LIGHTcount = 0
                            if label == "ac" and score > 0.9:
                                ACcount = ACcount + 1
                                if ACcount > trustVal and rptCtrl == 1:
                                    rptCtrl = 0
                                    print("You are pointing the Air Conditioner")
                                    acStat = not(acStat)
                                    if acStat == 1:
                                        x = requests.post(url, data=json.dumps({"command":"turn on the air conditioner"}), headers=headers)
                                    elif acStat == 0:
                                        x = requests.post(url, data=json.dumps({"command":"turn off the air conditioner"}), headers=headers)
                                    if x.status_code == 200:
                                        print('AC controlled successfully')
                                    ACcount = 0
                            if label == "tv" and score > 0.9:
                                TVcount = TVcount + 1
                                if TVcount > trustVal and rptCtrl == 1:
                                    rptCtrl = 0
                                    print("You are pointing the TV")
                                    tvStat = not(tvStat)
                                    if tvStat == 1:
                                        x = requests.post(url, data=json.dumps({"command":"turn on the TV"}), headers=headers)
                                    elif tvStat == 0:
                                        x = requests.post(url, data=json.dumps({"command":"turn off the TV"}), headers=headers)
                                    if x.status_code == 200:
                                        print('TV controlled successfully')
                                    
                                    TVcount = 0
                            if label == "other" and score > 0.9:
                                OTHERcount = OTHERcount + 1
                                if OTHERcount > 2:
                                    rptCtrl = 1
                                    LIGHTcount = 0
                                    ACcount = 0
                                    TVcount = 0
                                    OTHERcount = 0
                                

                            #print('%s: %.2f\t' % (label, score), end='')
                        #print('', flush=True)

                    elif "bounding_boxes" in res["result"].keys():
                        # print('Found %d bounding boxes (%d ms.)' % (len(res["result"]["bounding_boxes"]), res['timing']['dsp'] + res['timing']['classification']))
                        countPeople = len(res["result"]["bounding_boxes"])
                        # inferenceSpeed = res['timing']['classification']
                        for bb in res["result"]["bounding_boxes"]:
                            # print('\t%s (%.2f): x=%d y=%d w=%d h=%d' % (bb['label'], bb['value'], bb['x'], bb['y'], bb['width'], bb['height']))
                            img = cv2.rectangle(img, (bb['x'], bb['y']), (bb['x'] + bb['width'], bb['y'] + bb['height']), (0, 0, 255), 2)
                        
                    ret, buffer = cv2.imencode('.jpg', img)
                    buffer = cv2.cvtColor(ret, cv2.COLOR_BGR2RGB)

                    #/////////////////////////////////////////////////////////////

                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

                    next_frame = now() + 100
                    
            finally:
                if (runner):
                    runner.stop()


def get_inference_speed():
    while True:
        # print(inferenceSpeed)
        yield "data:" + str(inferenceSpeed) + "\n\n"
        time.sleep(0.1)

def get_people():
    while True:
        # print(countPeople)
        yield "data:" + str(countPeople) + "\n\n"
        time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/inference_speed')
def inference_speed():
	return Response(get_inference_speed(), mimetype= 'text/event-stream')

@app.route('/people_counter')
def people_counter():
	return Response(get_people(), mimetype= 'text/event-stream')


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=4912, debug=True)