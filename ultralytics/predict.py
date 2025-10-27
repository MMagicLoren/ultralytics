from ultralytics import YOLO

if __name__ == "__main__":
    source = r"E:\datasets\coco128-depth\val\images"
    model = YOLO(r"E:\WorkSpace\ultralytics\ultralytics\runs\depth\train3\weights\best.pt")

    # ============================================================================
    # 方式1: 在 model.predict() 中直接指定 depth_vis_mode 和 depth_alpha
    # ============================================================================
    
    # 默认模式: 叠加显示 (overlay, alpha=0.7)
    # results = model.predict(source=source, save=True)
    
    # 只保存深度图 (不叠加原图)
    # results = model.predict(source=source, save=True, imgsz=640, depth_vis_mode="depth_only")
    
    # 原图和深度图并排显示
    results = model.predict(source=source, save=True, imgsz=640, depth_vis_mode="side_by_side")
    
    # 叠加显示,自定义透明度
    # results = model.predict(source=source, save=True, imgsz=640, depth_vis_mode="overlay", depth_alpha=0.7)
    
    
    # ============================================================================
    # 方式2: 手动控制每个结果的可视化 (更灵活)
    # ============================================================================
    
    # results = model.predict(source=source, save=False)  # 不自动保存
    
    # # 为每个结果保存不同模式
    # for i, result in enumerate(results[:3]):  # 只处理前3张图片
    #     # 只保存深度图
    #     result.plot(depth_vis_mode="depth_only", save=True, filename=f"depth_only_{i}.jpg")
        
    #     # 并排显示
    #     result.plot(depth_vis_mode="side_by_side", save=True, filename=f"side_by_side_{i}.jpg")
        
    #     # 叠加显示 (alpha=0.5)
    #     result.plot(depth_vis_mode="overlay", depth_alpha=0.5, save=True, filename=f"overlay_{i}.jpg")