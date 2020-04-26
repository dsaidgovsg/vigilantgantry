# -*- coding: utf-8 -*-

"""
~~~~~~~~~~~~~~~
This file containts helper functions that supports the pipepline.


"""
import cv2


def xyxy2xywh(x):
    return x[0], x[1], x[2] - x[0], x[3] - x[1]


def xywh2xyxy(x):
    return [x[0], x[1], x[0] + x[2], x[1] + x[3]]


def get_centriod(x):
    x1, y1, x2, y2 = xywh2xyxy(x)
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def if_point_intersect_with_rect(rect, pt):
    return (
        pt[0] > rect[0][0]
        and pt[0] < rect[1][0]
        and pt[1] > rect[0][1]
        and pt[1] < rect[1][1]
    )


def get_frame_inside_roi(frame_inside_roi, bbox):
    f_min_x, f_min_y, f_max_x, f_max_y = bbox
    h, w = f_max_y - f_min_y, f_max_x - f_min_x
    frame_inside_roi_x = frame_inside_roi[:, :, ::-1]
    frame_inside_roi_y = frame_inside_roi_x[
        f_min_y : f_min_y + h, f_min_x : f_min_x + w
    ]
    return frame_inside_roi_y


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
    return (
        person_detect_intercept_boundary[0][0] - person_detect_roi_boundary[0][0],
        person_detect_intercept_boundary[0][1] - person_detect_roi_boundary[0][1],
        person_detect_intercept_boundary[1][0] - person_detect_roi_boundary[0][0],
        person_detect_intercept_boundary[1][1] - person_detect_roi_boundary[0][1],
    )


def display_bbox(frame, bbox, text, color="green"):
    color = (0, 255, 0)
    if color == "red":
        color = (0, 0, 255)
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
