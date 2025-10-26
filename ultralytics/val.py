from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"E:\WorkSpace\ultralytics\ultralytics\runs\depth\train\weights\best.pt")
    model.val(data="coco128-depth.yaml", imgsz=640, batch=8, device="0")