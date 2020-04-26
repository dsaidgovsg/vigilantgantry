# -*- coding: utf-8 -*-

"""
~~~~~~~~~~~~~~~
This video processor module ingestes video streams from either RTSP, webcam or video file and performs face segmentation.
"""
import cv2
import colorsys
import numpy as np
import threading

# from utils.vis_utils import display_bbox


def run_pipeline(
    frame,
    person_detector,
    face_detector,
    face_segmentor,
    face_seg_threshold_value,
    person_detect_roi_boundary,
    person_detect_intercept_boundary,
    gantry_id,
):
    intercept_zone = get_intercept_zone(
        person_detect_roi_boundary, person_detect_intercept_boundary
    )
    cv2.rectangle(
        frame,
        (intercept_zone[0], intercept_zone[1]),
        (intercept_zone[2], intercept_zone[3]),
        (255, 255, 255),
        1,
    )
    candidates = person_detector.get_human_bbox(frame, "person")
    for candidate in candidates:
        bbox = np.array(candidate).astype(int)
        if (bbox > 1).all():
            if if_point_intersect_with_rect(
                person_detect_intercept_boundary, get_centriod_x1y1x2y2_bbox(bbox)
            ):
                run_heuristic(
                    frame,
                    bbox,
                    face_seg_threshold_value,
                    face_detector,
                    face_segmentor,
                    gantry_id,
                )
                cv2.circle(frame, get_centriod_x1y1x2y2_bbox(bbox), 10, (0, 0, 255), -1)
    return frame


def run_heuristic(
    frame, bbox, threshold_value, face_detector, face_segmentor, gantry_id
):
    frame_inside_roi = get_frame_inside_roi(frame, xywhTOx1y1x2y2_bbox(bbox))
    boxes = face_detector.get_face_bbox(frame_inside_roi)
    threshold_set, _ = face_segmentor.get_segmentation_value(
        frame_inside_roi, xyxy2xywh(boxes[0])
    )
    if threshold_set > threshold_value:
        frame = display_bbox(frame, bbox, "OK", change_color=None)
        output = push_output_result(gantry_id, 1)
    else:
        frame = display_bbox(frame, bbox, "COVERED", 70, change_color=None)
        output = push_output_result(gantry_id, 0)
    print(output)
    return output


def xyxy2xywh(x):
    return x[0], x[1], x[2] - x[0], x[3] - x[1]


def xywhTOx1y1x2y2_bbox(x):
    return [x[0], x[1], x[0] + x[2], x[1] + x[3]]


def get_centriod_x1y1x2y2_bbox(x):
    x1, y1, x2, y2 = xywhTOx1y1x2y2_bbox(x)
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def if_point_intersect_with_rect(rect, pt):
    return pt[0] > rect[0] and pt[0] < rect[2] and pt[1] > rect[1] and pt[1] < rect[3]


def get_frame_inside_roi(frame_inside_roi, bbox):
    f_min_x, f_min_y, f_max_x, f_max_y = xywhTOx1y1x2y2_bbox(bbox)
    h, w = f_max_y - f_min_y, f_max_x - f_min_x
    frame_inside_roi = frame_inside_roi[:, :, ::-1]
    frame_inside_roi = frame_inside_roi[f_min_y : f_min_y + h, f_min_x : f_min_x + w]
    return frame_inside_roi


def push_output_result(gantry_id, verify):
    import uuid
    import time

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


def get_intercept_zone(person_detect_roi_boundary, person_detect_intercept_boundary):
    return [
        person_detect_intercept_boundary[0] - person_detect_roi_boundary[0][0],
        person_detect_intercept_boundary[1] - person_detect_roi_boundary[0][1],
        person_detect_intercept_boundary[2] - person_detect_roi_boundary[0][0],
        person_detect_intercept_boundary[3] - person_detect_roi_boundary[0][1],
    ]


def create_unique_color_uchar(tag, hue_step=0.41):
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255 * r), int(255 * g), int(255 * b)


def create_unique_color_float(tag, hue_step=0.41):
    h, v = (tag * hue_step) % 1, 1.0 - (int(tag * hue_step) % 4) / 5.0
    r, g, b = colorsys.hsv_to_rgb(h, 1.0, v)
    return r, g, b


def display_bbox(frame, bbox, text, color_id=1, change_color=None):
    color = create_unique_color_uchar(color_id)
    x1, y1, w, h = bbox
    p1 = (int(x1), int(y1))
    p2 = (int(x1 + w), int(y1 + h))
    cv2.rectangle(frame, p1, p2, color)

    l1 = (bbox[0], bbox[1] - 10)
    l2 = (bbox[2] + bbox[0], bbox[1])
    t1 = (bbox[0], bbox[1] - 3)
    cv2.rectangle(frame, l1, l2, color, cv2.FILLED)

    if sum(color) / 3 < 97:
        text_colour = (255, 255, 255)
    else:
        text_colour = (0, 0, 0)

    cv2.putText(
        frame, text, t1, cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_colour, 1, cv2.LINE_AA
    )

    return frame
