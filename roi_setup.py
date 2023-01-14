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