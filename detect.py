import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/exp/weights/best.pt')
    model.predict(source='images',
                  save=True,
                  show=False,
                  imgsz=640,
                  device='0',
                  )

