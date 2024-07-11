import threading
import time
import cv2
import numpy as np
import torch
import yaml
from torchvision import transforms
import asyncio
import aiohttp

from face_alignment.alignment import norm_crop

from face_detection.scrfd.detector import SCRFD

from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import compare_encodings, read_features

from face_tracking.tracker.byte_tracker import BYTETracker
from face_tracking.tracker.visualize import plot_tracking

from Flask.id import frameIDs,SearchingCond,ResetID
from Flask.car import RobotControl
from Flask.pantilt import PanTiltMoving
from Flask.mobilefeed import MobileFeed
from Flask.notification import SendNotifications
from Flask.endpoints import url_chs_id,url_chs_nm,url_R_feed


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Face detector
detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")
# detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5n-face.pt")

# Face recognizer
recognizer = iresnet_inference(
    model_name="r100", path="face_recognition/arcface/weights/arcface_r100.pth", device=device
)


id_face_mapping = {}
name_id_mapping = {}
real_time_ids={}


chosen_id_bymobile=None # remember to set this to none at the end, would be sent to mobile first so he can chooose from the current ids, so it avoids choosing unavailable id and making an error.

# Load precomputed face features and names
images_names, images_embs = read_features(feature_path="./datasets/face_features/feature")

# Data mapping for tracking information
data_mapping = {
    "raw_image": [],
    "tracking_ids": [],
    "detection_bboxes": [],
    "detection_landmarks": [],
    "tracking_bboxes": []
}


def load_config(file_name):
    """
    Load a YAML configuration file.

    Args:
        file_name (str): The path to the YAML configuration file.

    Returns:
        dict: The loaded configuration as a dictionary.
    """
    with open(file_name, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

# coding to measure distance 
KNOWN_DISTANCE = 53 # centimeter
# width of face in the real world or Object Plane
KNOWN_WIDTH = 15  # centimeter

# focal length finder function
def focal_length(measured_distance, real_width, width_in_rf_image):

    focal_length_value = (width_in_rf_image * measured_distance) / real_width
    return focal_length_value

# distance estimation function
def distance_finder(focal_length, real_face_width, face_width_in_frame):

    distancee = (real_face_width * focal_length) / face_width_in_frame
    return distancee

# face detector function
def Cal_Face_width(image):
    face_width = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml").detectMultiScale(gray_image, 1.3, 5)
    for (x, y, h, w) in faces:
        face_width = w
    return face_width

ref_image = cv2.imread("ref_image.jpg")
ref_image_face_width = Cal_Face_width(ref_image)
focal_length_found = focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, ref_image_face_width)


def process_tracking(frame, detector, tracker, args, frame_id, fps):
    """
    Process tracking for a frame.

    Args:
        frame: The input frame.
        detector: The face detector.
        tracker: The object tracker.
        args (dict): Tracking configuration parameters.
        frame_id (int): The frame ID.
        fps (float): Frames per second.

    Returns:
        numpy.ndarray: The processed tracking image.
    """

    # Face detection and tracking
    outputs, img_info, bboxes, landmarks = detector.detect_tracking(image=frame)


    if outputs is None or len(outputs) == 0:
        return img_info["raw_img"]

    tracking_tlwhs = []
    tracking_ids = []
    tracking_scores = []
    tracking_bboxes = []

    height = img_info["height"]
    width= img_info["width"]
    tlwh=None
    # print(height,width)

    online_targets = tracker.update(outputs, [img_info["height"], img_info["width"]], (128, 128))
    for t in online_targets:
        tlwh = t.tlwh
        tid = t.track_id
        vertical = tlwh[2] / tlwh[3] > args["aspect_ratio_thresh"]
        if tlwh[2] * tlwh[3] > args["min_box_area"] and not vertical:
            x1, y1, w, h = tlwh
            tracking_bboxes.append([x1, y1, x1 + w, y1 + h])
            tracking_tlwhs.append(tlwh)
            tracking_scores.append(t.score)
            tracking_ids.append(tid)

    tracking_image = plot_tracking(
        img_info["raw_img"],
        tracking_tlwhs,
        tracking_ids,
        cards=id_face_mapping,
        frame_id=frame_id + 1,
        fps=fps,
    )

    data_mapping["raw_image"] = img_info["raw_img"]
    data_mapping["detection_bboxes"] = bboxes
    data_mapping["detection_landmarks"] = landmarks
    data_mapping["tracking_ids"] = tracking_ids
    data_mapping["tracking_bboxes"] = tracking_bboxes

    # print(id_face_mapping,name_id_mapping)
    return tracking_image


@torch.no_grad()
def get_feature(face_image):
    """
    Extract features from a face image.

    Args:
        face_image: The input face image.

    Returns:
        numpy.ndarray: The extracted features.
    """
    face_preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((112, 112)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Convert to RGB
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    # Preprocess image (BGR)
    face_image = face_preprocess(face_image).unsqueeze(0).to(device)

    # Inference to get feature
    emb_img_face = recognizer(face_image).cpu().numpy()

    # Convert to array
    images_emb = emb_img_face / np.linalg.norm(emb_img_face)

    return images_emb


def recognition(face_image):
    """
    Recognize a face image.

    Args:
        face_image: The input face image.

    Returns:
        tuple: A tuple containing the recognition score and name.
    """
    # Get feature from face
    query_emb = get_feature(face_image)

    score, id_min = compare_encodings(query_emb, images_embs)
    name = images_names[id_min]
    score = score[0]

    return score, name


def mapping_bbox(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (tuple): The first bounding box (x_min, y_min, x_max, y_max).
        box2 (tuple): The second bounding box (x_min, y_min, x_max, y_max).

    Returns:
        float: The IoU score.
    """
    # Calculate the intersection area
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])

    intersection_area = max(0, x_max_inter - x_min_inter + 1) * max(
        0, y_max_inter - y_min_inter + 1
    )

    # Calculate the area of each bounding box
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate the union area
    union_area = area_box1 + area_box2 - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou


def tracking(detector, args):
    """
    Face tracking in a separate thread.

    Args:
        detector: The face detector.
        args (dict): Tracking configuration parameters.
    """
    
    # Initialize variables for measuring frame rate
    start_time = time.time_ns()
    frame_count = 0
    fps = -1

    # Initialize a tracker and a timer
    tracker = BYTETracker(args=args, frame_rate=30)
    frame_id = 0

    cap = cv2.VideoCapture(0)    
    # cap = cv2.VideoCapture(url_R_feed)

    while True:
        _, img = cap.read()
        # img=cv2.flip(img, 0)

        tracking_image = process_tracking(img, detector, tracker, args, frame_id, fps)

        # Calculate and display the frame rate
        frame_count += 1
        if frame_count >= 30:
            fps = 1e9 * frame_count / (time.time_ns() - start_time)
            frame_count = 0
            start_time = time.time_ns()

        cv2.imshow("Face Recognition", tracking_image)

        # asyncio.run(MobileFeed(tracking_image))

        # Check for user exit input
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
        
    cap.release()
    cv2.destroyAllWindows()

is_found=False
def recognize():
    general_id=None
    # global SearchingCond
    # global chosen_id_bymobile
    global is_found
    asyncio.run(PanTiltMoving.order(0,45))
    threshold = 5
    prev_xDelta = None
    prev_yDelta = None

    """Face recognition in a separate thread."""
    while True:
        raw_image = data_mapping["raw_image"]
        detection_landmarks = data_mapping["detection_landmarks"]
        detection_bboxes = data_mapping["detection_bboxes"]
        tracking_ids = data_mapping["tracking_ids"]
        tracking_bboxes = data_mapping["tracking_bboxes"]
        is_known_face=False
        for i in range(len(tracking_bboxes)):
            for j in range(len(detection_bboxes)):
                mapping_score = mapping_bbox(box1=tracking_bboxes[i], box2=detection_bboxes[j])
                if mapping_score > 0.9:
                    face_alignment = norm_crop(img=raw_image, landmark=detection_landmarks[j])

                    score, name = recognition(face_image=face_alignment)
                    if name is not None:
                        if score < 0.25:
                            caption = [f"UN_KNOWN_{tracking_ids[i]}",'non','UN_KNOWN']
                        else:
                            caption =[name,round(score*100,2),'KNOWN']
                    
                        if caption[0] in name_id_mapping:  # Check if the name already has an ID
                            face_id = name_id_mapping[caption[0]]
                        else:  # If the name is new, assign a new ID
                            name_id_mapping[caption[0]] = len(name_id_mapping) + 1
                            face_id = name_id_mapping[caption[0]]

                        face_width_in_frame = tracking_bboxes[i][2] - tracking_bboxes[i][0]
                        face_height_in_frame = tracking_bboxes[i][3] - tracking_bboxes[i][1]

                        Distance = distance_finder(focal_length_found, KNOWN_WIDTH, face_width_in_frame)

                        face_centerX= tracking_bboxes[i][0] + (face_width_in_frame/2)
                        face_centerY = tracking_bboxes[i][1] +(face_height_in_frame/2)

                        deltaX=int(face_centerX-(640//2))
                        deltaY=int(face_centerY-(480//2))

                        id_face_mapping[tracking_ids[i]]=[caption[0], caption[1], face_id, round(Distance,2), deltaX,deltaY,caption[2]]

                        # Priority choosing for known faces only
                        min_id=float('inf')
                        for trackid in tracking_ids:
                            real_time_ids[trackid] = id_face_mapping.get(trackid, ['Unknown', 'Unknown', float('inf'), 0, 0, 0, 'Unknown'])
                            if  real_time_ids[trackid][6]!='UN_KNOWN' and real_time_ids[trackid][2]<min_id :
                                min_id=real_time_ids[trackid][2]
                                general_id=trackid
                                is_known_face=True

                        print(real_time_ids)

                        current_ids = [data[2] for data in real_time_ids.values()]
                        print("Current IDs in frame: ", current_ids)

                        # Post current_ids to Flask Station
                        asyncio.run(frameIDs(current_ids))


                        if is_known_face and chosen_id_bymobile is None:
                            xDelta=real_time_ids[general_id][4]
                            yDelta=real_time_ids[general_id][5]

                            # if prev_xDelta is None or prev_yDelta is None or abs(xDelta - prev_xDelta) > threshold or abs(yDelta - prev_yDelta) > threshold:
                                # asyncio.run(PanTiltMoving.move(xDelta,yDelta))
                                # prev_xDelta, prev_yDelta = xDelta, yDelta
                            print('min_id: ',min_id,', general_id: ',general_id)


                        elif isinstance(chosen_id_bymobile, int): #as chosen id could be not none but 's'
                            chosen_general_id = None
                            for track_id, real_time_data in real_time_ids.items():
                                if chosen_id_bymobile == real_time_data[2]:
                                    chosen_general_id = track_id
                                    break

                            if chosen_general_id is not None:
                                xDelta = real_time_ids[chosen_general_id][4]
                                yDelta = real_time_ids[chosen_general_id][5]
                                zTrack = real_time_ids[chosen_general_id][3]

                                if prev_xDelta is None or prev_yDelta is None or abs(xDelta - prev_xDelta) > threshold or abs(yDelta - prev_yDelta) > threshold:
                                    asyncio.run(PanTiltMoving.ytracking(xDelta,yDelta))
                                    prev_xDelta, prev_yDelta = xDelta, yDelta

                                asyncio.run(RobotControl.attack(xDelta, zTrack))
                                print('chosen_id_bymobile: ',chosen_id_bymobile,', chosen_general_id: ', chosen_general_id)
                            else:
                                print(f"Chosen ID {chosen_id_bymobile} not found in real_time_ids")
                        print("-----------------------------------------")

                    detection_bboxes = np.delete(detection_bboxes, j, axis=0)
                    detection_landmarks = np.delete(detection_landmarks, j, axis=0)
                    real_time_ids.clear()

                    break

        if SearchingCond:
            asyncio.run(SearchingAlgorithm())
            # pass

    # cv2.destroyAllWindows()

async def SearchingAlgorithm(): # need to be edited
    global SearchingCond
    print('Start Process of Searching')
    while SearchingCond:
        await asyncio.sleep(1)
        pan=-90
        tilt=45
        print('pan= -90')
        # await PanTiltMoving.order(pan,tilt)
        await asyncio.sleep(1)

        while pan<90 and data_mapping["tracking_bboxes"]== [] and SearchingCond:
            pan=pan+45 
            print('pan++ =',pan)
            # await PanTiltMoving.order(pan,tilt)
            await asyncio.sleep(2)

        if data_mapping["tracking_bboxes"]!= []:
            print("is found")
            await SendNotifications('f') # send notification
            await ResetID() # to stop fetching id / executing search algo.
            return

        elif not SearchingCond:
            print("process is killed")
            return

        await RobotControl.rotate(180)
        await asyncio.sleep(4)
        
        while pan>-90 and data_mapping["tracking_bboxes"]== [] and SearchingCond:
            pan=pan-45 
            print('pan-- =',pan)
            # await PanTiltMoving.order(pan,tilt)
            await asyncio.sleep(2)

        if data_mapping["tracking_bboxes"]!= []:
            print("is found")
            await SendNotifications('f') # send notification
            await ResetID()
            return
        
        elif not SearchingCond:
            print("process is killed")
            return

        await RobotControl.rotate(180)

        await asyncio.sleep(2)
        print("Searching Process finished without finding any faces")
        # await PanTiltMoving.order(0,45)
        await SendNotifications('x') # send notification
        await ResetID()
        SearchingCond=False
        return 

async def fetch_chosen_id():
    global chosen_id_bymobile,SearchingCond,is_found
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url_chs_id) as response:
                if response.status == 200:
                    data = await response.json()
                    if data['id']=='s':
                        SearchingCond=True
                    else:
                        SearchingCond=False
                        chosen_id_bymobile = int(data['id'])
                        if chosen_id_bymobile==0:
                            chosen_id_bymobile=None #at the end of attack, ai would post (0) to /chosen_id
                        else:
                            print("Fetched chosen ID:", chosen_id_bymobile)
                else:
                    # print(f"Failed to fetch chosen ID: {response.status}")
                    pass
    except Exception as e:
        print(f"Error fetching chosen ID: {e}")

async def continuously_fetch_chosen_id(interval=2):
    while True:
        await fetch_chosen_id()
        await asyncio.sleep(interval)

def main():
    """Main function to start face tracking and recognition threads."""
    file_name = "./face_tracking/config/config_tracking.yaml"
    config_tracking = load_config(file_name)

    # Start tracking thread
    thread_track = threading.Thread(
        target=tracking,
        args=(detector,config_tracking,),
    )
    thread_track.start()

    # Start recognition thread
    thread_recognize = threading.Thread(target=recognize)
    thread_recognize.start()

    # Start the asyncio loop for continuous fetching of chosen ID
    loop = asyncio.get_event_loop()
    loop.create_task(continuously_fetch_chosen_id())

    # Run the loop in a separate thread
    loop_thread = threading.Thread(target=loop.run_forever)
    loop_thread.start()


if __name__ == "__main__":
    main()
