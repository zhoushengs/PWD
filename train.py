import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from ultralytics import RTDETR

# Load a model
 # build a new model from YAML
# model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# Train the model
if __name__ == '__main__':
    model = YOLO("ultralytics/cfg/models/11/yolo11.yaml")
    #odel = YOLO('weights/yolov5s.pt')
    #model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-resnet50.yaml')

    model.train(data="cfg/datasets/dota8.yaml",
                workers=1,
                lr0=0.001,
                lrf=0.005,
                device='0',
                batch=2,
                epochs=100,
                imgsz=640,
                amp=False)

