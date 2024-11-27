import cv2

# 视频文件路径q
video_path = r'D:\kend\WorkProject\Hk_Tracker\data\videos\palace.mp4'
# video_path = ('/home/lyh/work/ByteTrack-main/YOLOX_outputs/yolox_x_mix_det/track_vis/2024_11_25_16_37_11/K05_20241122150150-20241122150350_1.mp4')

# 使用OpenCV打开视频
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 显示帧q
    # print(frame.shape)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()