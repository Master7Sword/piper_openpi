import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    # ----------------------------------------------------
    # 1. 发现所有连接的 RealSense 设备
    # ----------------------------------------------------
    context = rs.context()
    devices = context.query_devices()
    serial_numbers = []
    for dev in devices:
        serial = dev.get_info(rs.camera_info.serial_number)
        serial_numbers.append(serial)

    if not serial_numbers:
        print("错误：未找到 RealSense 设备！")
        return

    print(f"找到 {len(serial_numbers)} 台设备:")
    for serial in serial_numbers:
        print(f"  - {serial}")

    # ----------------------------------------------------
    # 2. 为每台设备创建和配置 Pipeline
    # ----------------------------------------------------
    pipelines = {}
    configs = {}
    aligns = {}
    
    # 定义通用的流配置
    width, height, fps = 640, 480, 60

    for serial in serial_numbers:
        pipeline = rs.pipeline(context) # 使用同一个 context
        config = rs.config()
        
        # 关键：告诉 config 只使用这个序列号的设备
        config.enable_device(serial) 
        
        # 配置流
        try:
            # config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            
            pipelines[serial] = pipeline
            configs[serial] = config
            
            # 每个 pipeline 都需要自己的 align 对象
            aligns[serial] = rs.align(rs.stream.color) 
        except Exception as e:
            print(f"警告：无法为设备 {serial} 配置流（{width}x{height} @ {fps}）。")
            print(f"  错误: {e}")
            # 如果配置失败，从列表中移除
            if serial in serial_numbers: serial_numbers.remove(serial)


    # ----------------------------------------------------
    # 3. 启动所有配置成功的 pipelines
    # ----------------------------------------------------
    active_serials = [] # 只保留成功启动的设备
    try:
        for serial in serial_numbers:
            print(f"正在启动设备: {serial}...")
            try:
                pipelines[serial].start(configs[serial])
                active_serials.append(serial) # 添加到活动列表
                print(f"设备 {serial} 启动成功。")
            except Exception as e:
                print(f"错误：无法启动设备 {serial}。{e}")
                # 清理失败的 pipeline
                if serial in pipelines: del pipelines[serial]
                if serial in configs: del configs[serial]
                if serial in aligns: del aligns[serial]
        
        if not active_serials:
            print("没有设备成功启动。")
            return

        print(f"\n成功启动 {len(active_serials)} 台设备。按 'q' 或 'ESC' 退出...")

        # ----------------------------------------------------
        # 4. 主循环：从所有设备读取和显示
        # ----------------------------------------------------
        while True:
            # 存储所有摄像头的拼接图像
            all_device_images = []
            
            # 遍历所有成功启动的设备
            for serial in active_serials:
                pipeline = pipelines[serial]
                align = aligns[serial]
                
                # ----------------------------------------------------
                # ** 错误修正点 **
                # ----------------------------------------------------
                # 1. 解包 (success, frames) 元组
                success, frames = pipeline.try_wait_for_frames() 
                
                # 2. 检查布尔值 success
                if not success:
                    continue # 如果没有新帧，跳过此设备
                # ----------------------------------------------------

                # 4. 将深度帧对齐到彩色帧 (现在 'frames' 变量是正确的 frameset)
                aligned_frames = align.process(frames)
                # depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                # if not depth_frame or not color_frame:
                #     continue

                # 5. 将图像转换为 numpy 数组
                # depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # 6. 深度图可视化处理
                # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                # 7. 拼接单台设备的图像 (Color | Depth)
                # if serial[-4:] == '0835':
                #     # rotate 180 degrees for device 0835
                #     color_image = cv2.rotate(color_image, cv2.ROTATE_180)
                #     depth_colormap = cv2.rotate(depth_colormap, cv2.ROTATE_180)

                # 8. (可选) 在画面上添加设备信息
                # 只显示序列号后4位以示区分
                cv2.putText(color_image, f"Dev: {serial[-4:]}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # images_single_device = np.hstack((color_image, depth_colormap))
                images_single_device = color_image # 只显示彩色图像
                all_device_images.append(images_single_device)

            # 9. 拼接所有设备的图像
            if not all_device_images:
                continue # 如果所有设备都没有帧，则跳过

            # 垂直拼接所有设备的图像
            # Cam1: [Color | Depth]
            # Cam2: [Color | Depth]
            # Cam3: [Color | Depth]
            final_image = np.hstack(all_device_images)

            # 10. 使用 OpenCV 显示
            cv2.imshow('Multi-RealSense View', final_image)
            
            # 按 'q' 或 ESC 退出
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                break

    except KeyboardInterrupt:
        print("\n检测到键盘中断，正在退出...")
        for serial in active_serials:
            if serial in pipelines:
                pipelines[serial].stop()
        cv2.destroyAllWindows()
        print("已清理资源。")
        return

    finally:
        # 11. 停止所有 pipelines
        print("\n正在停止所有设备...")
        for serial in active_serials:
            if serial in pipelines:
                pipelines[serial].stop()
        cv2.destroyAllWindows()
        print("已清理资源。")

if __name__ == "__main__":
    main()