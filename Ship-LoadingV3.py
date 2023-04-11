# ----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# ----------------------------------------------------#

import torch
import os
import sys
import cv2
import numpy as np
import math
from  MobileNet_Line_Detection.mlsd_pytorch.cfg.default import  get_cfg_defaults
from MobileNet_Line_Detection.models.mbv2_mlsd_large import MobileV2_MLSD_Large
from MobileNet_Line_Detection.utils import pred_lines

# from BoatDetection.yolo import YOLO
from yolov5BoatDetection.yolo import YOLO
import time

import cv2
import numpy as np
from PIL import Image

from deeplab import DeeplabV3

if __name__ == "__main__":
    # -------------------------------------------------------------------------#
    #   如果想要修改对应种类的颜色，到generate函数里修改self.colors即可
    # -------------------------------------------------------------------------#
    deeplab = DeeplabV3()
    yolo = YOLO()
    # ----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    # ----------------------------------------------------------------------------------------------------------#
    mode = "predict"
    # mode = "video"
    # ----------------------------------------------------------------------------------------------------------#
    #   video_path用于指定视频的路径，当video_path=0时表示检测摄像头
    #   想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path表示视频保存的路径，当video_save_path=""时表示不保存
    #   想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps用于保存的视频的fps
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    # ----------------------------------------------------------------------------------------------------------#
    video_path = "img2/8.MOV"

    video_save_path = ""
    video_fps = 25.0
    # -------------------------------------------------------------------------#
    #   test_interval用于指定测量fps的时候，图片检测的次数
    #   理论上test_interval越大，fps越准确。
    # -------------------------------------------------------------------------#
    test_interval = 100
    # -------------------------------------------------------------------------#
    #   dir_origin_path指定了用于检测的图片的文件夹路径
    #   dir_save_path指定了检测完图片的保存路径
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    # -------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path = "img_out/"

    print("相机主点坐标为：")
    uc = 2319
    vc = 1565
    print(uc, vc)

    print("相机高度为：（单位为cm）")
    Hcam = 424 #真实船舶相机高度
    # Hcam = 9 #模型船舶相机高度
    print(Hcam)

    if mode == "predict":
        '''
        predict.py有几个注意点
        1、该代码无法直接进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
        具体流程可以参考get_miou_prediction.py，在get_miou_prediction.py即实现了遍历。
        2、如果想要保存，利用r_image.save("img.jpg")即可保存。
        3、如果想要原图和分割图不混合，可以把blend参数设置成False。
        4、如果想根据mask获取对应的区域，可以参考detect_image函数中，利用预测结果绘图的部分，判断每一个像素点的种类，然后根据种类获取对应的部分。
        seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
        for c in range(self.num_classes):
            seg_img[:, :, 0] += ((pr == c)*( self.colors[c][0] )).astype('uint8')
            seg_img[:, :, 1] += ((pr == c)*( self.colors[c][1] )).astype('uint8')
            seg_img[:, :, 2] += ((pr == c)*( self.colors[c][2] )).astype('uint8')
        '''

        img = "img/1.jpg"
        img1 = cv2.imread(img)
        image1 = Image.open(img)
        image = Image.open(img)
        start = time.time()

        r_image, dfre = yolo.detect_image(image)
        print(dfre)
        for i in range(len(dfre)):
            boat = image1.crop(dfre[i])
            r_image = deeplab.detect_image(boat)
            # image.paste(l_image, (dfre[i][0], dfre[i][1], dfre[i][2], dfre[i][3]))
            r_image.save(str(i + 5) + '.jpg')
        # r_image = deeplab.detect_image(image)


        # print("所用时间为：")
        # print(start1 - start)

        # r_image.save("2.jpg")

            r_image = cv2.cvtColor(np.asarray(r_image),cv2.COLOR_RGB2BGR)

        #二值化
            img_gray = cv2.cvtColor(r_image, cv2.COLOR_RGB2GRAY)
            ret, binary = cv2.threshold(img_gray, 125, 255, cv2.THRESH_BINARY)
            cv2.imwrite("4.jpg", binary)

        #去除不需要的轮廓
            img_mask = np.zeros(img_gray.shape[:2], dtype="uint8")
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
            mask = cv2.drawContours(img_mask, contours, 0, 255, cv2.FILLED)
        #-----------------------------------------------------------------------------------------------------------
            # binary = np.array(binary)
            binary = np.array(img_mask)
            h, w = binary.shape

            temp = int(w / 50)
        # start = time.time()
            result = []
            result1 = []
            for k in range(100 - 1):
                x = []
                p = []
                for t in range(h - 1):
                    y = int(temp / 2)
                    if (binary[t][y * (k + 1)]) == 0:
                        if (binary[t + 1][(k + 1) * y]) == 255:
                            x.append(t+dfre[i][1])
                            x.append(((k + 1) * y) + dfre[i][0])
                    if (binary[t][y * (k + 1)]) == 255:
                        if (binary[t + 1][(k + 1) * y]) == 0:
                            p.append(t+dfre[i][1])
                            p.append(((k + 1) * y) + dfre[i][0])
                if len(x) != 0:
                    result.append(x)
                if len(p) != 0:
                    result1.append(p)

                print(result)
                print(result1)



            x_x = []
            y_y = []
            for r in result:
                x_x.append(r[1])
                y_y.append(r[0])
                point1 = (r[1], r[0])
                cv2.circle(img1, point1, 8, (0, 255, 255), -1)
            x_2 = []
            y_2 = []

            for r in result1:
                x_2.append(r[1])
                y_2.append(r[0])
                point1 = (r[1], r[0])
                cv2.circle(img1, point1, 8, (0, 255, 255), -1)

            def Fun(x, a1, a2):  # 定义拟合函数形式
                return a1 * x + a2

            x_x = np.array(x_x)
            y_y = np.array(y_y)
            para = np.polyfit(x_x, y_y, deg=1)
            print(para)
            y_fitted = Fun(x_x, para[0], para[1])

            x_2 = np.array(x_2)
            y_2 = np.array(y_2)
            para1 = np.polyfit(x_2, y_2, deg=1)
            print(para1)
            y_fitted1 = Fun(x_2, para1[0], para1[1])

            cv2.line(img1, (int(x_x[0]), int(y_fitted[0])), (int(x_x[len(x_x) - 1]), int(y_fitted[len(y_fitted) - 1])),
                     (0, 0, 255), 2, 16)
            cv2.line(img1, (int(x_2[0]), int(y_fitted1[0])), (int(x_2[len(x_x) - 1]), int(y_fitted1[len(y_fitted) - 1])),
                     (0, 0, 255), 2, 16)
            # end = time.time()
            # print(end - start)


            print("-------------------------------------------------")


            print("确定船舶干舷位置：")
            middle_top_x = (x_x[0] + x_x[len(x_x) - 1])/2
            middle_top_y = (y_fitted[0] + y_fitted[len(y_fitted) - 1])/2
            print(middle_top_x, middle_top_y)

            k = (y_fitted1[len(y_fitted) - 1] - y_fitted1[0]) * 1.0 / (x_2[len(x_x) - 1] - x_2[0])
            a = k
            b = -1.0
            c = y_fitted1[0] - k * x_2[0] #直线公式
            p_foot_x = int((b * b * middle_top_x - a * b * middle_top_y - a * c) / (a * a + b * b))
            p_foot_y = int((a * a * middle_top_y - a * b * middle_top_x - b * c) / (a * a + b * b))
            print("垂足坐标为：")
            print(p_foot_x, p_foot_y)


            y = k * middle_top_x + c
            print("竖直坐标为：")
            print(middle_top_x, y)

            print("两者间的夹角为：")
            the = math.atan((p_foot_x - middle_top_x) / (p_foot_y - middle_top_y))
            print(the)


            hobjp = int(y) - int(middle_top_y)
            print("船舶像素干舷为：")
            print(hobjp)
            hcamp = int(y) - int(vc)

            Hobj = (hobjp * Hcam / hcamp) * math.cos(the)
            print("干舷结果为：(单位为cm)")
            print(Hobj)


            # k1 = (x[f][3] - x[f][1]) * 1.0 / (x[f][2] - x[f][0])
            # k2 = -1/k1 #其中一条直线斜率
            # b2 = middle_top_y * 1.0 - middle_top_x * k2 * 1.0 #其中一条直线
            #
            # k3 = k  # 斜率存在操作
            # b3 = c
            #
            # x_x = (b3 - b2) * 1.0 / (k2 - k3)
            # y_y = k2 * x_x * 1.0 + b2 * 1.0
            # print("过甲板线干舷点作垂线与吃水线的交点为：")
            # print(x_x, y_y)



            #画图
            point = (int(middle_top_x), int(middle_top_y))
            point1 = (p_foot_x, p_foot_y)
            point2 = (int(middle_top_x), int(y))
            # cv2.circle(img1, point, 5, (0, 0, 255), -1)
            # cv2.circle(img1, point1, 8, (0, 255, 255), -1)
            # cv2.circle(img1, point2, 8, (255, 255, 255), -1)


            # end = time.time()
            #
            # print(end-start)



        cv2.imwrite("1.jpg", img1)


        print("-------------------------------------------------")



    elif mode == "video":
        timeef = 4
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        sum = 0

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        Hobj_result = []
        while (True):
            sum = sum + 1
            # t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            if ref == True and (sum%timeef==0):
                t1 = time.time()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 转变成Image
                # print(frame)
                frame = Image.fromarray(np.uint8(frame))
                image1 = frame
                # 进行检测
                # frame = np.array(deeplab.detect_image(frame))
                r_image, dfre = yolo.detect_image(frame)

                if(len(dfre)) == 0:
                    continue

                frame = np.array(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                for i in range(len(dfre)):
                    boat = image1.crop(dfre[i])
                    r_image = deeplab.detect_image(boat)

                    r_image = cv2.cvtColor(np.asarray(r_image), cv2.COLOR_RGB2BGR)

                    img_gray = cv2.cvtColor(r_image, cv2.COLOR_RGB2GRAY)
                    ret, binary = cv2.threshold(img_gray, 125, 255, cv2.THRESH_BINARY)

                    #只保留最大的轮廓
                    img_mask = np.zeros(img_gray.shape[:2], dtype="uint8")
                    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
                    mask = cv2.drawContours(img_mask, contours, 0, 255, cv2.FILLED)

                    binary = np.array(img_mask)
                    h, w = binary.shape

                # img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                # ret, binary = cv2.threshold(img_gray, 125, 255, cv2.THRESH_BINARY)
                # binary = np.array(binary)
                # h, w = binary.shape

                    temp = int(w / 20)
                    # start = time.time()
                    result = []
                    result1 = []
                    for k in range(40 - 1):
                        x = []
                        p = []
                        for t in range(h - 1):
                            y = int(temp / 2)
                            if (binary[t][y * (k + 1)]) == 0:
                                if (binary[t + 1][(k + 1) * y]) == 255:
                                    x.append(t + dfre[i][1])
                                    x.append(((k + 1) * y) + dfre[i][0])
                            if (binary[t][y * (k + 1)]) == 255:
                                if (binary[t + 1][(k + 1) * y]) == 0:
                                    p.append(t + dfre[i][1])
                                    p.append(((k + 1) * y) + dfre[i][0])
                        if len(x) != 0:
                            result.append(x)
                        if len(p) != 0:
                            result1.append(p)

                    print(result)
                    print(result1)

                    x_x = []
                    y_y = []
                    for r in result:
                        x_x.append(r[1])
                        y_y.append(r[0])
                        point1 = (r[1], r[0])
                        cv2.circle(frame, point1, 4, (0, 255, 255), -1)
                    x_2 = []
                    y_2 = []

                    for r in result1:
                        x_2.append(r[1])
                        y_2.append(r[0])
                        point1 = (r[1], r[0])
                        cv2.circle(frame, point1, 4, (0, 255, 255), -1)


                    def Fun(x, a1, a2):  # 定义拟合函数形式
                        return a1 * x + a2

                    if(len(x_x) or len(y_y)) == 0:
                        continue

                    x_x = np.array(x_x)
                    y_y = np.array(y_y)
                    para = np.polyfit(x_x, y_y, deg=1)
                    print(para)
                    y_fitted = Fun(x_x, para[0], para[1])

                    x_2 = np.array(x_2)
                    y_2 = np.array(y_2)
                    para1 = np.polyfit(x_2, y_2, deg=1)
                    print(para1)
                    y_fitted1 = Fun(x_2, para1[0], para1[1])

                    if (len(x_2) or len(y_2)) == 0:
                        continue

                    if (len(y_fitted1) or len(y_fitted)) == 0:
                        continue

                    # cv2.line(frame, (int(x_x[0]), int(y_fitted[0])), (int(x_x[len(x_x) - 1]), int(y_fitted[len(y_fitted) - 1])),
                    #          (0, 0, 255), 2, 8)
                    cv2.line(frame, (int(x_2[0]), int(y_fitted1[0])),
                             (int(x_2[len(x_x) - 1]), int(y_fitted1[len(y_fitted) - 1])),
                             (0, 0, 255), 2, 8)
                    # end = time.time()
                    # print(end - start)

                    print("-------------------------------------------------")

                    print("确定船舶干舷位置：")
                    middle_top_x = (x_x[0] + x_x[len(x_x) - 1]) / 2
                    middle_top_y = (y_fitted[0] + y_fitted[len(y_fitted) - 1]) / 2
                    print(middle_top_x, middle_top_y)

                    if(x_2[len(x_x) - 1] - x_2[0]) == 0:
                        continue

                    k = (y_fitted1[len(y_fitted) - 1] - y_fitted1[0]) * 1.0 / (x_2[len(x_x) - 1] - x_2[0])
                    a = k
                    b = -1.0
                    c = y_fitted1[0] - k * x_2[0]  # 直线公式
                    p_foot_x = int((b * b * middle_top_x - a * b * middle_top_y - a * c) / (a * a + b * b))
                    p_foot_y = int((a * a * middle_top_y - a * b * middle_top_x - b * c) / (a * a + b * b))
                    print("垂足坐标为：")
                    print(p_foot_x, p_foot_y)

                    y = k * middle_top_x + c
                    print("竖直坐标为：")
                    print(middle_top_x, y)

                    print("两者间的夹角为：")
                    the = math.atan((p_foot_x - middle_top_x) / (p_foot_y - middle_top_y))
                    print(the)

                    print("干舷结果为：(单位为cm)")
                    hobjp = int(y) - int(middle_top_y)
                    hcamp = int(y) - int(vc)

                    Hobj = (hobjp * Hcam / hcamp) * math.cos(the)
                    Hobj_result.append(Hobj)
                    print(Hobj)

                    # k1 = (x[f][3] - x[f][1]) * 1.0 / (x[f][2] - x[f][0])
                    # k2 = -1/k1 #其中一条直线斜率
                    # b2 = middle_top_y * 1.0 - middle_top_x * k2 * 1.0 #其中一条直线
                    #
                    # k3 = k  # 斜率存在操作
                    # b3 = c
                    #
                    # x_x = (b3 - b2) * 1.0 / (k2 - k3)
                    # y_y = k2 * x_x * 1.0 + b2 * 1.0
                    # print("过甲板线干舷点作垂线与吃水线的交点为：")
                    # print(x_x, y_y)

                    # 画图
                    point = (int(middle_top_x), int(middle_top_y))
                    point1 = (p_foot_x, p_foot_y)
                    point2 = (int(middle_top_x), int(y))
                    # cv2.circle(frame, point, 5, (0, 0, 255), -1)
                    # cv2.circle(frame, point1, 8, (0, 255, 255), -1)
                    # cv2.circle(frame, point2, 8, (255, 255, 255), -1)

                    end = time.time()

                    # print(end - start)


                    # RGBtoBGR满足opencv显示格式



                fps = (fps + (1. / (time.time() - t1))) / 2
                print("fps= %.2f" % (fps))
                frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("video", frame)
                c = cv2.waitKey(1) & 0xff
                if video_save_path != "":
                    out.write(frame)

                if c == 27:
                    capture.release()
                    break

        length = len(Hobj_result)
        sum = 0
        for i in range(len(Hobj_result)):
            sum = sum + Hobj_result[i]

        print("最终船舶干舷为：")
        print(sum/length)

        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open('img/street.jpg')
        tact_time = deeplab.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image = deeplab.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
