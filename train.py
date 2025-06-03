import warnings
from ultralytics import YOLO
from ultralytics import RTDETR

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # model = YOLO(r'./ultralytics/cfg/models/Eye-YOLO.yaml')
    model = YOLO(r'./ultralytics/cfg/models/11/yolo11.yamlEye-YOLO.yaml')
    # resume
    # model = YOLO(r'./runs/train/exp/weights/last.pt')

    model.train(data=r'./ultralytics/datasets/slit_smart.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                single_cls=False,
                batch=16,
                close_mosaic=10,
                workers=8,
                device='0',
                optimizer='SGD',
                amp=False,  
                pretrained=True,  
                project='./runs/train_slit_smart',
                name='exp',
                resume=False,
                plots=True,  
                save_period=-1,
                )
