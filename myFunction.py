import cv2
import numpy as np

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    # rho：线段以像素为单位的距离精度
    # theta : 像素以弧度为单位的角度精度(np.pi/180较为合适)
    # threshold : 霍夫平面累加的阈值
    # minLineLength : 线段最小长度(像素级)
    # maxLineGap : 最大允许断裂长度
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    return lines


def draw_lines(image, lines, color=[255, 255, 255], thickness=1):
    right_y_set = []
    right_x_set = []
    right_slope_set = []

    left_y_set = []
    left_x_set = []
    left_slope_set = []

    slope_min = 0.35  # 斜率低阈值
    slope_max = 10  # 斜率高阈值
    middle_x = image.shape[1] / 2  # 图像中线x坐标
    max_y = image.shape[0]  # 最大y坐标

    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)  # 拟合成直线
            slope = fit[0]  # 斜率

            if slope_min < np.absolute(slope) <= slope_max:

                # 将斜率大于0且线段X坐标在图像中线右边的点存为右边车道线
                if slope > 0 and x1 > middle_x and x2 > middle_x:
                    right_y_set.append(y1)
                    right_y_set.append(y2)
                    right_x_set.append(x1)
                    right_x_set.append(x2)
                    right_slope_set.append(slope)

                # 将斜率小于0且线段X坐标在图像中线左边的点存为左边车道线
                elif slope < 0 and x1 < middle_x and x2 < middle_x:
                    left_y_set.append(y1)
                    left_y_set.append(y2)
                    left_x_set.append(x1)
                    left_x_set.append(x2)
                    left_slope_set.append(slope)
    # 绘制右车道线
    fill_set = []
    if right_y_set:
        rindex = right_y_set.index(min(right_y_set))  # 最高点
        right_x_top = right_x_set[rindex]
        right_y_top = right_y_set[rindex]
        rslope = np.median(right_slope_set)
        # 根据斜率计算车道线与图片下方交点作为起点
        right_x_bottom = int(right_x_top + (max_y - right_y_top) / rslope)
        # 绘制线段
        cv2.line(image, (right_x_top, right_y_top), (right_x_bottom, max_y), color, thickness)
        fill_set.append([right_x_top, right_y_top])
        fill_set.append([right_x_bottom, max_y])
    # 绘制左车道线
    if left_y_set:
        lindex = left_y_set.index(min(left_y_set))  # 最高点
        left_x_top = left_x_set[lindex]
        left_y_top = left_y_set[lindex]
        lslope = np.median(left_slope_set)  # 计算平均值
        # 根据斜率计算车道线与图片下方交点作为起点
        left_x_bottom = int(left_x_top + (max_y - left_y_top) / lslope)

        # 绘制线段
        cv2.line(image, (left_x_bottom, max_y), (left_x_top, left_y_top), color, thickness)
        fill_set.append([left_x_bottom, max_y])
        fill_set.append([left_x_top, left_y_top])
    # print(fill_set)
    if (len(fill_set) == 4):
        cv2.fillConvexPoly(image, np.array(fill_set), [0, 255, 0])


def weighted_img(img, initial_img, a=1., b=1., c=0.):
    return cv2.addWeighted(initial_img, a, img, b, c)