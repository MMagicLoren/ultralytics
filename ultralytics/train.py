from ultralytics import YOLO


if __name__ == "__main__":
    # model = YOLO("yolov8n-seg.yaml")
    # results = model.train(
    #     data="coco128-seg.yaml", epochs=2, imgsz=640, batch=16, device="0", workers=0,)
    
    model = YOLO("yolov8n-depth.yaml").load("yolov8n-seg.pt")
    results = model.train(
        data="coco128-depth.yaml", 
        epochs=2, 
        imgsz=640, 
        batch=8, 
        device="0", 
        workers=0,
    )
