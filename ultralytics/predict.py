from ultralytics import YOLO

if __name__ == "__main__":
    source = "./images"
    model = YOLO("./runs/depth/train/weights/best.pt")

    # 默认模式: 叠加显示 (overlay, alpha=0.7)
    # results = model.predict(source=source, save=True)
    
    # 只保存深度图 (不叠加原图)
    # results = model.predict(source=source, save=True, imgsz=640, depth_vis_mode="depth_only")
    
    # 原图和深度图并排显示
    results = model.predict(source=source, save=True, imgsz=640, depth_vis_mode="side_by_side")
    
    # 叠加显示,自定义透明度
    # results = model.predict(source=source, save=True, imgsz=640, depth_vis_mode="overlay", depth_alpha=0.7)
    