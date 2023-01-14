import os
import cv2
import time

img_path = r'D:\Desktop\CGwork\CGwork\figure\out\out1'
# 随便从其中拿到一张图片来代表视频中图片的尺寸
img = cv2.imread(r'D:\Desktop\CGwork\CGwork\figure\out\out1\1.jpg')
imgInfo = img.shape
size = (imgInfo[1], imgInfo[0])
# 获得文件夹中图片的数量，从而进行循环生成视频文件
img_nums = len(os.listdir(img_path))
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
# 写入对象 1 file name 2 编码器 3 帧率 4 尺寸大小
videoWrite = cv2.VideoWriter(
os.path.join(img_path,r'D:\Desktop/', 'detect_result3.mp4'), fourcc, 30, size)

# 读取这个文件夹中的每一张图片（按照顺序）然后组合成视频，帧率是每秒 30 帧
for i in range(len(os.listdir(img_path))):
    filename = str(i+1) + ".jpg"
    filename = os.path.join(img_path, filename)
    # print(filename)
    img = cv2.imread(filename)
    videoWrite.write(img)

