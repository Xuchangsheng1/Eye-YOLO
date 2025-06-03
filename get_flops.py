from ultralytics.utils.torch_utils import get_flops
from ultralytics.utils.torch_utils import get_num_params
from ultralytics.utils.torch_utils import model_info
from ultralytics import YOLO
# model = './datasets/yolo11.yaml'

model = YOLO('./weights/last.pt')

flops = get_flops(model, imgsz=640)
# params = get_num_params(model)
# Model_Info = model_info(model, detailed=False, verbose=True, imgsz=640)
# print('flops', flops)
# print('params', params)
# print('Model_Info', Model_Info)
