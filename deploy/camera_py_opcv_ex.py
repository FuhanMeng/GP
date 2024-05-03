import cv2

# 打开摄像头
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # 读取视频帧
    ret, frame = cap.read()

    # 检查是否成功读取帧
    if not ret:
        print("Error: Could not read frame.")
        break

    # 显示当前帧
    cv2.imshow('Camera', frame)

    # 按下Esc键退出循环
    if cv2.waitKey(1) == 27:
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()