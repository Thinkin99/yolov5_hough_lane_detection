# yolov5_hough_lane_detection
A repo realize the detection of vehicles, pedestrians and lane lines during driving. ALL IN ONE

这学期做的一个大作业，基于YOLOv5和Hough变换实现了对行驶过程中车辆、行人以及车道线的检测。

[B站视频演示](https://www.bilibili.com/video/BV1JD4y1H7bR/?vd_source=fb58e5b322a71f814e90d5eebc7585cf)

# How to Use

主要的检测流程是：

 1. 选择一段你喜欢的路况视频，按帧分解为图片（提供视频帧分解程序`mp4tofigure.py`）在figure文件夹中有部分图片素材。
 2. 图片预处理，设定Canny高低阈值以及ROI标定（提供动态调整Canny高低阈值的辅助程序 `Canny_check.py`，交互式ROI标定的辅助程序`roi_setup.py`）
 3. 送入主程序`main.py`，进行目标检测和车道线检测。
 4. 检测结果为图片，可以转换成视频（提供导出视频程序`figure2video.py`）
 
# 一、实现效果

第一个是b站中国街景的驾车实拍视频，第二个是其他车道线检测里拿的视频素材。


![在这里插入图片描述](https://github.com/Thinkin99/yolov5_hough_lane_detection/blob/main/figure_md/result1.png)
![在这里插入图片描述](https://github.com/Thinkin99/yolov5_hough_lane_detection/blob/main/figure_md/result2.png)

# 二、环境配置
一个可以运行YOLOv5的python环境，与之前的repo：[yolov5_d435i_detection](https://github.com/Thinkin99/yolov5_d435i_detection)差不多。


# 三、基于YOLOv5的目标检测
这部分没啥特别的，就是套YOLOv5框架。不过由于原版的YOLOv5代码过于冗长，为后期与车道线检测的代码相结合，对YOLOv5源代码进行适当的简化封装，将检测过程基于一个封装的函数实现，如下所示。Canvas为返回的图像矩阵，class_id_list为检测到类的id列表，xyxy_list为检测框的角点位置像素坐标，conf_list为置信度列表。

```bash
canvas, class_id_list, xyxy_list, conf_list = model.detect(color_image)
```

选用YOLOv5s权重，其中前八类为为交通相关。
`['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light']`
效果如下
![在这里插入图片描述](https://github.com/Thinkin99/yolov5_hough_lane_detection/blob/main/figure_md/yolo.jpg)
# 四、基于Hough变换的车道线检测
基于Hough变换的车道线检测流程为：输入图片——Canny轮廓检测——设定ROI——Hough变换提取直线——结果信息绘制。
## 4.1 前置工作 Canny阈值设定
首先需要调整一下Canny的阈值，使得轮廓检测出来的车道线连续平滑。
这里提供一个可以动态调整Canny高低阈值的辅助程序 `Canny_check.py`
![在这里插入图片描述](https://github.com/Thinkin99/yolov5_hough_lane_detection/blob/main/figure_md/canny_check.png)


```bash
import cv2
para=[300,400]
def nothing(*arg):
    pass
cv2.namedWindow('Trackbar')
cv2.resizeWindow('Trackbar', 400, 100)
cv2.createTrackbar('low', 'Trackbar', para[0], 1000, nothing)
cv2.createTrackbar('high', 'Trackbar', para[1], 1000, nothing)

source=cv2.imread("test.jpg")
img=source.copy()
cv2.namedWindow("canny", 0)
while(1):
    img = source.copy()
    low = cv2.getTrackbarPos('low', 'Trackbar')
    high = cv2.getTrackbarPos('high', 'Trackbar')
    img = cv2.Canny(img, low, high)
    cv2.imshow("canny", img)
    cv2.waitKey(20)
```
## 4.2 前置工作 ROI标定
接着需要标定出车道线检测范围的ROI
这里提供一个交互式ROI标定的辅助程序`roi_setup.py`，按顺时针顺序左键进行像素标定，右键完成像素点连接，中键清除所有标记，最终可以导出标定结果。
![在这里插入图片描述](https://github.com/Thinkin99/yolov5_hough_lane_detection/blob/main/figure_md/roi.png)
![在这里插入图片描述](https://github.com/Thinkin99/yolov5_hough_lane_detection/blob/main/figure_md/roi2.png)

```bash
import cv2
import numpy as np

def mouse(event, x, y, flags, param):
    img1 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:

        x_list.append(x)
        y_list.append(y)
        # print(x_list)
        for i in range(len(x_list)):
            xy = "(%d,%d)" % (x_list[i], y_list[i])
            cv2.circle(img1, (x_list[i], y_list[i]), 1, (0, 0, 255), thickness=2)
            cv2.putText(img1, xy, (x_list[i], y_list[i]), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
        cv2.imshow("image", img1)  # 显示坐标
    if event == cv2.EVENT_RBUTTONDOWN:
        for i in range(len(x_list)-1):
            cv2.line(img1, [x_list[i],y_list[i]] ,[x_list[i+1],y_list[i+1]],(0,0,255),thickness=2)
        cv2.line(img1, [x_list[-1], y_list[-1]], [x_list[0], y_list[0]], (0, 0, 255), thickness=2)
        cv2.imshow("image", img1)
    if event == cv2.EVENT_MBUTTONDOWN:
        img1 = img.copy()
        cv2.imshow("image", img1)
        x_list.clear()
        y_list.clear()


def get_coordinate_by_click(img_path):
    global img
    img = cv2.imread(img_path)  # 图片路径
    cv2.namedWindow("image", 0)  # 设置窗口标题和大小
    cv2.setMouseCallback("image", mouse)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    print("x坐标:",x_list)
    print("y坐标:",y_list)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    x_list=[]
    y_list=[]
    img_path = r'./figure/2/1.jpg'
    get_coordinate_by_click(img_path)
```
## 4.3 Hough变换提取直线
按照之前标定的ROI进行Hough变换，检测到的直线绘制一个绿色的蒙板。
![在这里插入图片描述](https://github.com/Thinkin99/yolov5_hough_lane_detection/blob/main/figure_md/hough1.png)
![在这里插入图片描述](https://github.com/Thinkin99/yolov5_hough_lane_detection/blob/main/figure_md/hough2.png)
# 五、核心代码

```python
import numpy as np
import cv2
from myClass import YoloV5
from myFunction import weighted_img, draw_lines, hough_lines


if __name__ == '__main__':
    print("[INFO] 开始YoloV5模型加载")
    # YOLOV5模型配置文件(YAML格式)的路径 yolov5_yaml_path
    model = YoloV5(yolov5_yaml_path='config/yolov5s.yaml')
    print("[INFO] 完成YoloV5模型加载")
    low = 200 #Canny low
    high = 300 #Canny high
    rho = 1  # 霍夫像素单位
    theta = np.pi / 360  # 霍夫角度移动步长
    hof_threshold = 20  # 霍夫平面累加阈值threshold
    min_line_len = 10  # 线段最小长度
    max_line_gap = 20  # 最大允许断裂长度
    index = 0 #图片索引
    while True:
        index = index + 1
        # 一张一张图片进行检测按index进行索引
        path = r"./figure/1/" + str(index) + ".jpg"
        path_output = r"./out/2/" + str(index) + ".jpg"
        color_image = cv2.imread(path)
        lane_img=color_image.copy()
        edges = cv2.Canny(lane_img, low, high)
        mask = np.zeros_like(edges)
        # vertices = np.array( [[(554, 463), (733, 464), (1112, 654), (298, 671)]],dtype=np.int32)#素材2的ROI
        vertices = np.array( [[(757, 800), (1150, 800), (1556, 1064), (314, 1064)]],dtype=np.int32)#素材1的ROI
        cv2.fillPoly(mask, vertices, 255)#绿色蒙板绘制
        masked_edges = cv2.bitwise_and(edges, mask)  # 按位与
        line_image = np.zeros_like(lane_img)
        # 绘制车道线线段
        lines = hough_lines(masked_edges, rho, theta, hof_threshold, min_line_len, max_line_gap)
        draw_lines(line_image, lines, thickness=10)
        # YoloV5 目标检测
        canvas, class_id_list, xyxy_list, conf_list = model.detect(color_image)
        if xyxy_list:
            for i in range(len(xyxy_list)):
                ux = int((xyxy_list[i][0] + xyxy_list[i][2]) / 2)  # 计算像素坐标系的x
                uy = int((xyxy_list[i][1] + xyxy_list[i][3]) / 2)  # 计算像素坐标系的y
                cv2.circle(canvas, (ux, uy), 4, (255, 255, 255), 5)  # 标出中心点
                cv2.putText(canvas, str([ux, uy]), (ux + 20, uy + 10), 0, 1,
                            [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)  # 标出坐标
        canvas=weighted_img(canvas, line_image)
        # 可视化部分
        cv2.namedWindow("raw", 0)
        cv2.imshow('raw',color_image)
        cv2.namedWindow("line", 0)
        cv2.imshow('line',line_image)
        cv2.namedWindow('detection', 0)
        cv2.imshow('detection', canvas)
        # cv2.imwrite(path_output, canvas)#图片保存
        key = cv2.waitKey()
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

```
# 六、总结
因为是基于传统视觉方案，检测效果鲁棒性不高，容易受到光照、路面甚至是虚线、斑马线的影响。

后期有时间整个LaneNet的玩玩。
