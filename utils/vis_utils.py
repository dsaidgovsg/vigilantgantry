import colorsys
import cv2

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