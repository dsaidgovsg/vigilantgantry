import torch
from torchvision import transforms
from face_seg.nets.MobileNetV2_unet import MobileNetV2_unet
from face_det.vision.ssd.config.fd_config import define_img_size
from face_det.vision.ssd.mb_tiny_RFB_fd import (
    create_Mb_Tiny_RFB_fd,
    create_Mb_Tiny_RFB_fd_predictor,
)
from face_det.vision.utils.misc import Timer

pre_trained = "face_seg/checkpoints/model.pt"
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

def load_model():
    model = MobileNetV2_unet(None).to("cuda")
    state_dict = torch.load(pre_trained, map_location="cuda")
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()
input_img_size = 480
define_img_size(input_img_size)

label_path = "face_det/models/voc-model-labels.txt"
net_type = "mb_tiny_RFB_fd" #args.net_type
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
test_device = "cuda:0"
candidate_size = 1000 
threshold = 0.5 
model_path = "face_det/models/Mb_Tiny_RFB_FD_train_input_320.pth"
net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
net.load(model_path)
predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=test_device)
