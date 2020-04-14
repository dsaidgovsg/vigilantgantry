import os
import time
import uuid
import threading
from datetime import datetime
import PIL

import cv2
import torch
import requests
import numpy as np

from vidgear.gears import CamGear
from utils.obj_utils import get_human_bbox, load_yolov3_model
from utils.vis_utils import display_bbox
from utils.seg_utils import predictor, model, transforms

import logging
from standard_logs.logger import setup_logging

# logging essentials
setup_logging()
logger = logging.getLogger(name="face-seg:vid_utils.py")
logger = logging.getLogger(__name__)

# Endpoint url to post results to for face segmentation results
ENDPOINT = os.getenv("ENDPOINT", "http://localhost:5000/live_results")

# VideoStreamer object class for video streaming capabilities
class VideoStreamer:
    # init function
    def __init__(self, src=0, width=640, height=480):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.thread = None

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    # start streaming
    def start(self):
        if self.started:
            logger.info("VideoStreamer has started")
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    # update cap
    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    # read frame
    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return frame

    # stop streaming
    def stop(self):
        self.started = False
        self.thread.join()

    # exit stream
    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()

# checks if input is rtsp
def if_rtsp(x):
    return True if x[:4] == "rtsp" else False

# checks if input is a webcam (expects an int which represents the device id)
def if_webcam(x):
    return type(x) is int

# resizes the incoming image
def resize_img(frame, max_length=640):
    H, W = frame.shape[:2]
    if max(W, H) > max_length:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR

    if W > H:
        W_resize = max_length
        H_resize = int(H * max_length / W)
    else:
        H_resize = max_length
        W_resize = int(W * max_length / H)
    frame = cv2.resize(frame, (W_resize, H_resize), interpolation=interpolation)
    return frame, W_resize, H_resize

# Saves ROI, so that it can be used later to identifying ROI
def save_roi(roi, lane_no):
    now = datetime.now()
    current_time = now.strftime("%Y%m%d%H%M%S")
    with open(f"roi/{str(lane_no)}_{str(current_time)}.txt", "w") as f:
        f.write(str(roi))

# Loads ROI
def load_roi(fn):
    with open(fn, "r") as f:
        return eval(f.read())

# Prompts user to draw roi, and saves it
def return_roi(frame):
    roi_1 = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=False)
    roi_1 = xywh2x1y1x2y2(roi_1)
    save_roi(roi_1, 1)
    cv2.rectangle(frame, roi_1[0], roi_1[1], (0, 255, 0), 2)
    roi_2 = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=False)
    roi_2 = xywh2x1y1x2y2(roi_2)
    save_roi(roi_2, 2)
    cv2.rectangle(frame, roi_2[0], roi_2[1], (0, 255, 0), 2)
    cv2.destroyAllWindows()

    return roi_1, roi_2

# changes bounding box format from XYWH to X1Y1X2Y2
def xywhTOx1y1x2y2(x):
    return ((x[0], x[1]), (x[0] + x[2], x[1] + x[3]))

# changes bounding box format from X1Y1X2Y2 to XYWH 
def x1y1x2y2TOxywh(x):
    return (x[0][0], x[0][1], x[1][0] - x[0][0], x[1][1] - x[0][1])

# calculates bounding box area
def get_bbox_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

# calculates length ratio of bounding box to ROI
def get_roi_bbox_length_ratio(roi, bbox):
    return (bbox[2] - bbox[0]) / (roi[1][0] - roi[0][0])

# calculate area ratio of bounding box to ROI
def get_roi_bbox_area_ratio(roi, bbox):
    return get_bbox_area(bbox) / ((roi[1][0] - roi[0][0]) * (roi[1][1] - roi[0][1]))

# checks if bounding box intersects ROI
def if_inside_rectange(roi, bbox):
    return (
        bbox[0] >= roi[0][0]
        and bbox[1] >= roi[0][1]
        and bbox[2] <= roi[1][0]
        and bbox[3] <= roi[1][1]
    )

# calculate area of intersection between bounding box and ROI
def get_percent_intersection(roi, bbox):
    if if_inside_rectange(roi, bbox):
        return 1.0
    else:
        xA = max(roi[0][0], bbox[0])
        yA = max(roi[0][1], bbox[1])
        xB = min(roi[1][0], bbox[2])
        yB = min(roi[1][1], bbox[3])
        intsc_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        bbox_area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
        return intsc_area / bbox_area

# get the upper half of the bounding box
def get_upper_portion(bbox, percent):
    x1, y1, x2, y2 = bbox
    return x1, y1, x2, int(y2 * percent)

# checks if a point is in a rectangle
def insideRect(rect, pt):
    return pt[0] > rect[0] and pt[0] < rect[2] and pt[1] > rect[1] and pt[1] < rect[3]

# runs face segmentation, and also saves cropped image
def calculate_seg(image, face_coord):
    f_min_x, f_min_y, f_max_x, f_max_y = xywhTOx1y1x2y2_bbox(face_coord)
    h, w = f_max_y - f_min_y, f_max_x - f_min_x
    pil_img_c = image[f_min_y : f_min_y + h, f_min_x : f_min_x + w]
    pil_img = PIL.Image.fromarray(pil_img_c)
    now = datetime.now()
    
    try:
        pil_img.save('/app/sample/crop_img/' + str(now)+".jpg")
    except Exception as e:
        logger.info(e,flush=True)
        pass

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    torch_img_c = transform(pil_img)
    torch_img = torch_img_c.unsqueeze(0)
    torch_img_a = torch_img.to("cuda")
    logits = model(torch_img_a)
    mask = torch.max(logits, 1)[1]
    masked_img_c = mask.squeeze().data.cpu().numpy()
    masked_img = np.uint8(masked_img_c)
    masked_img_b = np.where(masked_img == 2, 1, masked_img)
    masked_img_b = cv2.resize(masked_img_b, (pil_img_c.shape[1], pil_img_c.shape[0]))
    image = display_seg(image, masked_img_b, face_coord)
    return (np.sum(masked_img_b) / (masked_img_b.shape[0] * masked_img_b.shape[1])), image

# display segmentation image 
def display_seg(image, masked_img_b, face_coord):
    f_min_x, f_min_y, f_max_x, f_max_y = xywhTOx1y1x2y2_bbox(face_coord)
    masked_img_b = np.where(masked_img_b == 1, 255, masked_img_b)
    masked_img_b = np.where(masked_img_b == 2, 255, masked_img_b)
    masked_img = cv2.cvtColor(masked_img_b, cv2.COLOR_GRAY2BGR)
    indices = np.where(masked_img == 255)
    masked_img[indices[0], indices[1], :] = [0, 255, 0]
    image[
        f_min_y : f_min_y + masked_img.shape[0], f_min_x : f_min_x + masked_img.shape[1]
    ] = masked_img
    return image

def xyxy2xywh(x):
    return x[0], x[1], x[2]-x[0], x[3]-x[1]

# process individual frame
def process_frame(cur_img, roi, lane_hitzone, gantry_id, model, threshold_value):
    try:
        cv2.rectangle(cur_img, roi[0], roi[1], (255, 255, 255))
        vis_img = cur_img[roi[0][1] : roi[1][1], roi[0][0] : roi[1][0]]
        human_candidates = get_human_bbox(vis_img, model, "person")
        vis_img, output1 = run_face_seg_engine(
                vis_img, roi, lane_hitzone, human_candidates, threshold_value, gantry_id=gantry_id,
            )
        cur_img[roi[0][1] : roi[1][1], roi[0][0] : roi[1][0]] = vis_img
                
        return cur_img, output1

    except Exception as e:
        logger.error(e)
    return cur_img, output1

# process entire video
def process_video(
    vid_path,
    roi,
    lane_hitzone,
    gantry_id,
    threshold_value,
    full_screen=False,
):

    if if_rtsp(str(vid_path)):
        vid = VideoStreamer(str(vid_path)).start()

    elif if_webcam(vid_path):
        vid = VideoStreamer(int(vid_path)).start()

    else:
        vid = CamGear(source=str(vid_path)).start()

    yolo_model = load_yolov3_model()

    while True:
        frame = vid.read()
        if frame is None:
            break
        frame, output = process_frame(
            frame, roi, lane_hitzone, gantry_id, yolo_model, threshold_value
        )
        cv2.imshow("Face Segmemtation", frame)
        if cv2.waitKey(1) == 27:
            break
    vid.stop()
    cv2.destroyAllWindows()

# run face segmentation
def run_face_seg_engine(
    img, roi, lane_hitzone, candidates, threshold_value, gantry_id, change_color=None
):
    lane1_hitzone = [
        lane_hitzone[0] - roi[0][0],
        lane_hitzone[1] - roi[0][1],
        lane_hitzone[2] - roi[0][0],
        lane_hitzone[3] - roi[0][1],
    ]
    cv2.rectangle(
        img,
        (lane1_hitzone[0], lane1_hitzone[1]),
        (lane1_hitzone[2], lane1_hitzone[3]),
        (255, 255, 255),
        1,
    )
    output1 = None

    try:
        for candidate in candidates:
            bbox = np.array(candidate).astype(int)
            x1, y1, x2, y2 = bbox
            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                cv2.circle(img, get_centriod_x1y1x2y2_bbox(bbox), 10, (0, 0, 255), -1)
                if insideRect(lane1_hitzone, get_centriod_x1y1x2y2_bbox(bbox)):
                    output1 = push_output(img, bbox, threshold_value, gantry_id)
                    requests.post(ENDPOINT,json=output1)

        return img, output1
    except:
        return img, output1

def get_centriod_x1y1x2y2_bbox(x):
    x1, y1, x2, y2 = xywhTOx1y1x2y2_bbox(x)
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def insideRect(rect,pt): 
    return (pt[0] > rect[0] and pt[0] < rect[2] and pt[1] > rect[1] and pt[1] < rect[3])

def push_output(img, bbox, threshold_value, gantry_id):
    output = output_module(gantry_id, 0)
    try:
        f_min_x, f_min_y, f_max_x, f_max_y = xywhTOx1y1x2y2_bbox(bbox)
        h, w = f_max_y - f_min_y, f_max_x - f_min_x
        imgx = img[:, :, ::-1]
        imgx = imgx[f_min_y : f_min_y + h, f_min_x : f_min_x + w]
        threshold_set, _ = display_face(imgx)
        if threshold_set > threshold_value:
            img = display_bbox(img, bbox, "OK", change_color=None)
            output = output_module(gantry_id, 1)
        else:
            img = display_bbox(img, bbox, "COVERED", 70, change_color=None)
            output = output_module(gantry_id, 0)
    except Exception as e:
        pass
    logger.info(output)
    return output

def xywhTOx1y1x2y2_bbox(x):
    return [x[0], x[1], x[0] + x[2], x[1] + x[3]]

def display_face(vis_img):
    candidate_size = 1000
    threshold = 0.5
    boxes, labels, probs = predictor.predict(vis_img, candidate_size / 2, threshold)
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        box[0], box[1], box[2], box[3] = (
            box[0],
            box[1],
            box[2],
            box[3],
        )
        box = np.array(box).astype(int)
        rst, vis_img = calculate_seg(vis_img, xyxy2xywh(box))
        return rst, vis_img

def output_module(gantry_id, verify):
    uuid_id = str(uuid.uuid4())
    millis = int(round(time.time() * 1000))
    output_json = {
        "request_id": uuid_id,
        "gantry_id": gantry_id,
        "timestamp": millis,
        "pass": verify,
        "check_type": "face_segmentation",
    }
    return output_json