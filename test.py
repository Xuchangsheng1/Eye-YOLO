import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('./runs/train/exp/weights/best.pt')

    model.val(data='./ultralytics/datasets/xiaomi_smart.yaml',
              imgsz=640,
              batch=16,
              split='test',
              workers=10,
              device='0',
              project='./ultralytics/runs/test',
              name='exp',
              )

