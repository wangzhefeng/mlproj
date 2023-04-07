# -*- coding: utf-8 -*-


# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-03-10
# * Version     : 0.1.031000
# * Description : 算法：基于Dlib进行人脸检测与标记是指对于任意输入的
# *               目标图像通过算法策略对其进行搜索来检测其中是否包含有
# *               人脸特征的图像区域
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


"""
算法：基于Dlib进行人脸检测与标记是指对于任意输入的目标图像通过算法策略对其进行搜索来检测其中是否包含有人脸特征的图像区域。
"""

# python libraries
import os
import dlib
from skimage import io
import cv2
print(cv.__version__)


# 加载 dlib 检测器
detector = dlib.get_frontal_face_detector()
img = io.imread("images/girl1.jpg")

# 人脸检测
dets = detector(img, 1)
print(f"检测到的人脸数目: {len(dets)}.")
for d in dets:
    #使用OpenCV在原图上标出人脸位置
    left_top=(dlib.rectangle.left(d),dlib.rectangle.top(d))#左上方坐标
    right_bottom=(dlib.rectangle.right(d),dlib.rectangle.bottom(d))#右下方坐标
    cv2.rectangle(img,left_top,right_bottom,(0,255,0),2,cv2.LINE_AA)#画矩形
    cv2.imshow("img",cv2.cvtColor(img,cv2.COLOR_RGB2BGR))#转成BGR格式显示
cv2.waitKey(0)
cv2.destroyAllWindows()



"""
算法：基于Dlib的人脸检测与识别是通过多级级联的回归树进行关键点的回归。
文献：Kazemi, V. , &  Sullivan, J. . (2014). One Millisecond Face Alignment with an Ensemble of Regression Trees. IEEE Conference on Computer Vision & Pattern Recognition. IEEE.
Zhou, E. ,  Fan, H. ,  Cao, Z. ,  Jiang, Y. , &  Qi, Y. . (2013). Extensive Facial Landmark Localization with Coarse-to-Fine Convolutional Network Cascade. IEEE International Conference on Computer Vision Workshops. IEEE.
链接：https://pypi.org/project/dlib/#files
http://dlib.net/files/
https://cmake.org/download/
https://cmake.org/files/
https://www.boost.org/
"""


import cv2
import dlib
from skimage import io
#使用特征提取器get_frontal_face_detector
detector=dlib.get_frontal_face_detector()
#dlib的68点模型，使用作者训练好的特征预测器
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#图片所在路径
img=io.imread("C:/Users/xpp/Desktop/Lena.png")
#生成dlib的图像窗口
win=dlib.image_window()
win.clear_overlay()
win.set_image(img)
#特征提取器的实例化
dets=detector(img, 1)
print("人脸数：",len(dets))
for k, d in enumerate(dets):
    print("第",k+1,"个人脸d的坐标：",
          "left:",d.left(),
          "right:",d.right(),
          "top:",d.top(),
          "bottom:",d.bottom())
    width=d.right()-d.left()
    heigth=d.bottom()-d.top()
    print('人脸面积为：',(width*heigth))
    #利用预测器预测
    shape=predictor(img, d)
    #标出68个点的位置
    for i in range(68):
        cv2.circle(img,(shape.part(i).x,shape.part(i).y),4,(203,192,255),-1,8)
        cv2.putText(img,str(i),(shape.part(i).x,shape.part(i).y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
    #显示一下处理的图片，然后销毁窗口
    cv2.imshow('result',img)
    cv2.waitKey(0)


# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

