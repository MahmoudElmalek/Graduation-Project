from flask import Flask, Response, request, jsonify
import cv2
import requests, time
# import numpy as np
import subprocess

from Flask.endpoints import url_R_feed, url_R_pt, url_R_mv, url_R_op, url_mob_feed
from add_persons import case


app = Flask(__name__)

opt = 'm'  # Default option is Mobile
RaspberryFeed=url_R_feed
AiFeed= url_mob_feed # Station Flask Server, where AI post processed feed..
current_video_url = RaspberryFeed
cap = None
chosen_id = None
frame_ids = None
new_name=None
new_id=None
casee=case

class Controller:
    def __init__(self, script_name):
        self.script_name = script_name
        self.process = None

    def start(self):
        if self.process is None or self.process.poll() is not None:
            self.process = subprocess.Popen(["python", self.script_name])
            print(f"{self.script_name} process started.")
        else:
            print(f"{self.script_name} process is already running.")

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()  # or use self.process.kill() if terminate() is not effective
            self.process.wait()
            print(f"{self.script_name} process stopped.")
        else:
            print(f"{self.script_name} process is not running.")

    def is_running(self):
        return self.process and self.process.poll() is None

# Create manager instances for both scripts
recognize_manager = Controller("recognize.py")
addperson_manager = Controller("add_person.py")
    
def RunRecognize():
    if not recognize_manager.is_running():
        print("Starting recognize.py now.")
        recognize_manager.start()
    else:
        print("recognize.py is already running.")

def KillRecognize():
    if recognize_manager.is_running():
        print("Stopping recognize.py now.")
        recognize_manager.stop()
        print("recognize.py stopped.")
    else:
        print("recognize.py is not running.")

def RunAddPerson():
    if not addperson_manager.is_running():
        print("Starting add_person.py now.")
        addperson_manager.start()
    else:
        print("add_person.py is already running.")

def KillAddPerson():
    if addperson_manager.is_running():
        print("Stopping add_person.py now.")
        addperson_manager.stop()
        print("add_person.py stopped.")
    else:
        print("addperson.py is not running.")

def release_resources():
    global cap
    if cap is not None:
        cap.release()
        cap = None

latest_frame = None  # To store the latest frame from AI processed feed

def generate_frames():
    global opt, current_video_url, cap, latest_frame
    
    while opt in ['m', 'g']:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(current_video_url)
        
        ret, frame = cap.read()
        if not ret:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    if cap is not None:
        cap.release()

def generate_ai_frames(): # takes frames from vid_back and is called in video_feed so it is shown there
    global latest_frame
    while opt == 'a' or 'n':
        if latest_frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
        time.sleep(0.1)

@app.route('/video_feed') # where to read feeds from station, mobile reads, and addperson reads
def video_feed():
    if opt in ['m', 'g']:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif opt == 'a' or 'n':
        return Response(generate_ai_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response("Video feed disabled", mimetype='text/plain')

@app.route('/vid_back', methods=['POST']) # we post here feed, either from Ai output or mobile camera
def vid_back(): 
    global latest_frame
    latest_frame = request.data
    return '', 204  # No Content

@app.route('/set_opt', methods=['POST'])
def set_opt():
    global opt, current_video_url
    data = request.get_json()

    if data is None or 'opt' not in data:
        return jsonify({"error": "Invalid JSON or missing 'opt' key"}), 400

    # Reading New Option
    new_opt = data['opt']

    # Exclude mistaken options
    if new_opt not in ['m', 'a', 'g', 'n']:
        return jsonify({"error": "Invalid value for 'opt'"}), 400
    

    # Adding New Person
    if new_opt=='n':
        print('Adding new person..')

        if opt =='a':
            KillRecognize()

        release_resources()   # works for disabling Raspberry too (m and g cond.)
        time.sleep(2)

        # Run Addperson code
        RunAddPerson()

    # Choosing Mobile
    elif new_opt in ['m', 'g']:

        if opt=='a':
            print('Releasing resources of AI feed and switching to Raspberry feed...')
            KillRecognize()

        elif opt=='n':
            KillAddPerson()

        release_resources()
        time.sleep(2)  # to ensure the endpoint of Raspberry is released and free for direct connection to the station..
        current_video_url = RaspberryFeed  # Raspberry Flask Server

    # Choosing AI
    elif new_opt == 'a':

        if opt=='n':
            KillAddPerson()

        elif opt in ['m', 'g']:
            print("Releasing resources of Raspberry and switching to AI feed...")

        release_resources()
        time.sleep(2)  # to ensure the endpoint of Raspberry feed is released for the model to connect to..
        
        # Run Recognize
        RunRecognize()
        time.sleep(15)  # Wait for 15 seconds till the AI model works and post its processed feed..
        current_video_url = AiFeed  
    
    opt = new_opt

    # Forward the option to the Raspberry Pi server
    if opt != 'n':
        try:
            response = requests.post(url_R_op, json={'o': opt})
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return jsonify({"error": str(e)}), 500

    print("Option set to ", opt)
    return jsonify({"message": f"Option set to {opt}"}), 200

@app.route('/pan_tilt', methods=['POST'])
def pan_tilt():
    global opt
    if opt != 'm':
        return jsonify({"error": "Pan-Tilt commands are disabled"}), 403

    data = request.get_json()
    if data is None or 'pan' not in data or 'tilt' not in data:
        return jsonify({"error": "Invalid JSON or missing 'pan'/'tilt' keys"}), 400

    pan_angle = data['pan']
    tilt_angle = data['tilt']

    # Forward the pan and tilt values to the Raspberry Pi server
    try:
        response = requests.post(url_R_pt, json={'pan': pan_angle, 'tilt': tilt_angle})
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"message": "Pan-Tilt command forwarded"}), 200

@app.route('/car_mv', methods=['POST'])
def car_mv():
    global opt
    if opt not in ['m', 'a']:
        return jsonify({"error": "car_mv commands are disabled"}), 403

    data = request.get_json()
    if data is None or 'r' not in data or 'l' not in data:
        return jsonify({"error": "Invalid JSON or missing 'r'/'l' keys"}), 400

    rightspeed = data['r']
    leftspeed = data['l']

    print(rightspeed)

    # Forward the values to the Raspberry Pi server
    try:
        response = requests.post(url_R_mv, json={'r': rightspeed, 'l': leftspeed})
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"message": "car_mv command forwarded"}), 200

@app.route('/frame_ids', methods=['POST'])
def frame_idss():
    global frame_ids
    data = request.get_json()
    frame_ids = data.get('ids', []) 
    print("Received frame IDs:", frame_ids) 
    return jsonify({"status": "success", "received_ids": frame_ids})

@app.route('/get_frame_ids', methods=['GET'])
def get_frame_ids():
    if frame_ids is not None:
        return jsonify({"ids": frame_ids}), 200
    else:
        return jsonify({"message": "No IDs shown yet"}), 404

@app.route('/chosen_id', methods=['POST'])
def chosen_idd():
    global chosen_id
    data = request.json
    chosen_id = data['id']
    print("Received chosen ID:", chosen_id)
    return jsonify({"message": "Chosen ID received"}), 200

@app.route('/get_chosen_id', methods=['GET'])
def get_chosen_id():
    if chosen_id is not None:
        return jsonify({"id": chosen_id}), 200
    else:
        return jsonify({"message": "No ID received yet"}), 404

@app.route('/notify', methods=['GET'])
def notification():
    global casee  # c,n,s
    if casee is not None:
        return jsonify({"case": casee}), 200
    else:
        return jsonify({"message": "No notifications received yet"}), 404

@app.route('/new_person', methods=['POST'])
def add_person():
    global new_name,new_id

    data = request.get_json()

    if data is None or 'id' not in data or 'name' not in data:
        return jsonify({"error": "Invalid JSON or missing 'pan'/'tilt' keys"}), 400

    new_id = data['id']
    new_name = data['name']

    print("Received NewPerson: ", new_id,': ',new_name)
    return jsonify({"message": "NewPerson received"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
    manager = Controller()

