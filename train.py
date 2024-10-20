from ultralytics import YOLO

# Load a model
# on the basis of pre-training
# model = YOLO("/headless/Desktop/yolov11/yolov5nu.pt")

# a new model
model = YOLO(model=r'/headless/Desktop/yolov11/ultralytics/ultralytics/cfg/models/11/yolo11.yaml')

# Train the model
train_results = model.train(
    data=r'/headless/Desktop/yolov11/dataset/AbnormalDriverBehaviouryolov11/data.yaml',  # path to dataset YAML
    epochs=512,  # number of training epochs
    batch=32,
    imgsz=640,  # training image size
    device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model