import torch
import numpy as np
from yolo_v3.util import load_classes, write_results
from yolo_v3.darknet import Darknet
from yolo_v3.preprocess import prep_image
from torch.autograd import Variable


class args_ori():
    bs = 1
    nms_thresh = 0.4
    cfgfile = 'yolo_v3/cfg/yolov3.cfg'
    weightsfile = 'yolo_v3/yolov3.weights'
    reso = '416'
    scales='1,2,3'
    confidence = 0.7

num_classes = 80
classes = load_classes(
    "yolo_v3/data/coco.names"
)    
scales = args_ori.scales
batch_size = int(args_ori.bs)
confidence = float(args_ori.confidence)
nms_thesh = float(args_ori.nms_thresh)
CUDA = torch.cuda.is_available()

def load_yolov3_model():
    start = 0

    classes = load_classes('yolo_v3/data/coco.names')
    model = Darknet(args_ori.cfgfile)
    model.load_weights(args_ori.weightsfile)

    model.net_info["height"] = args_ori.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    model.eval()
    return model


class args_tiny():
    bs = 1
    nms_thresh = 0.4
    cfgfile = "yolo_v3/cfg/yolov3.cfg"
    weightsfile = "yolo_v3/yolov3.weights"
    reso = '416'
    scales='1,2,3'
    confidence = 0.7

scales = args_tiny.scales
batch_size = int(args_tiny.bs)
confidence = float(args_tiny.confidence)
nms_thesh = float(args_tiny.nms_thresh)
CUDA = torch.cuda.is_available()

num_classes = 80
classes = load_classes(
    "yolo_v3/data/coco.names"
)


def inference(images, model, classes):
    """
    images: numpy array
    model: yolo model
    return: human bboxs, scores
    """
    start = 0
    classes = classes

    imlist = [images]
    inp_dim = int(model.net_info["height"])
    batches = list(map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
    im_batches = [x[0] for x in batches]
    orig_ims = [x[1] for x in batches]
    im_dim_list = [x[2] for x in batches]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

    if CUDA:
        im_dim_list = im_dim_list.cuda()

    for batch in im_batches:
        if CUDA:
            batch = batch.cuda()
        with torch.no_grad():
            prediction = model(Variable(batch), CUDA)

        output = write_results(
            prediction, confidence, num_classes, nms=True, nms_conf=nms_thesh
        )

        if CUDA:
            torch.cuda.synchronize()

    try:
        output
    except NameError:
        print("No detections were made")
        exit()

    im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())
    scaling_factor = torch.min(inp_dim / im_dim_list, 1)[0].view(-1, 1)
    output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
    output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2
    output[:, 1:5] /= scaling_factor

    human_candidates = []
    scores = []
    im_id_list = []
    for i in range(len(output)):
        item = output[i]
        im_id = item[-1]
        im_id_list.append(im_id)
        if int(im_id) in [0, 7]:
            bbox = item[1:5].cpu().numpy()
            bbox = [round(i, 2) for i in list(bbox)]
            score = item[5]
            human_candidates.append(bbox)
            scores.append(score)
    scores = np.expand_dims(np.array(scores), 0)
    human_candidates = np.array(human_candidates)
    return human_candidates, scores, im_id_list

def xyhw_(self, tensor: torch.tensor):
    tensor_xy = torch.zeros_like(tensor)

    tensor_xy[:, :2] = tensor[:, :2] / self.grid_num - 0.5 * tensor[:, 2:4]
    tensor_xy[:, 2:4] = tensor[:, :2] / self.grid_num + 0.5 * tensor[:, 2:4]
    tensor_xy[:, 5:7] = tensor[:, 5:7] / self.grid_num - 0.5 * tensor[:, 7:9]
    tensor_xy[:, 7:9] = tensor[:, 5:7] / self.grid_num + 0.5 * tensor[:, 7:9]
    tensor_xy[:, 4], tensor_xy[:, 9] = tensor[:, 4], tensor[:, 9]

    return tensor_xy

def get_human_bbox(image, model, img_class):
    classes = load_classes(
    "yolo_v3/data/coco.names"
)    
    bboxs, probs, cls = inference(image, model, classes)
    human_candidates = []
    for b, c in zip(bboxs, cls):
        if str(classes[int(c)]) == img_class:
            x1, y1, x2, y2 = b
            w, h = x2 - x1, y2 - y1
            human_candidates.append([x1, y1, w, h])
    return human_candidates