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
    # ----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    # ----------------------------------------------------------------------------------------------------------#
    mode = "predict"
    # ----------------------------------------------------------------------------------------------------------#
    #   video_path用于指定视频的路径，当video_path=0时表示检测摄像头
    #   想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path表示视频保存的路径，当video_save_path=""时表示不保存
    #   想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps用于保存的视频的fps
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    # ----------------------------------------------------------------------------------------------------------#
    video_path = 0
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
    # Hcam = 424 #真实船舶相机高度
    Hcam = 9 #模型船舶相机高度
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

        image = Image.open(img)
        start = time.time()

        r_image = deeplab.detect_image(image)

        start1 = time.time()
        print("所用时间为：")
        print(start1 - start)

        r_image.save("2.jpg")

        r_image = cv2.cvtColor(np.asarray(r_image),cv2.COLOR_RGB2BGR)

        #二值化
        img_gray = cv2.cvtColor(r_image, cv2.COLOR_RGB2GRAY)
        ret, binary = cv2.threshold(img_gray, 125, 255, cv2.THRESH_BINARY)
        binary = cv2.merge((binary,binary,binary))


        print("-------------------------------------------------")

        cfg = get_cfg_defaults()
        current_dir = os.path.dirname(__file__)
        if current_dir == "":
            current_dir = "./"
        model_path = current_dir + '/' + 'MobileNet_Line_Detection/workdir/pretrained_models/mobilev2_mlsd_large_512_bsize24/best.pth'
        model = MobileV2_MLSD_Large(cfg).cuda().eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)


        img1 = cv2.imread(img)

        # lines = pred_lines(binary, model, [256, 256], 0.2, 50)
        lines = pred_lines(binary, model, [256, 256], 0.09, 5)
        print(lines)

        x = []
        j = 0
        for l in lines:
            if abs(l[0] - l[2]) > 500:
            # if abs(l[1] - l[3]) < 50:
                x.append(l)
                j = j + 1
        temp = 0
        h = []
        for i in range(j):
            if x[i][1] > temp:
                index = i
                temp = x[i][1]
        for i in range(j):
            h.append(x[i][1])
        print(h)
        sorted_id = sorted(range(len(h)), key=lambda k: h[k], reverse=True)
        print(sorted_id)

        h = 20
        a = []
        for s in sorted_id:
            # print(s)
            k = x[index][1] - x[s][1]

            if k >= h:
                a.append(s)
        print(a)
        f = a[0]
        cv2.line(img1, (int(x[f][0]), int(x[f][1])), (int(x[f][2]), int(x[f][3])), (0, 0, 255), 2, 16)
        print(x[f])

        cv2.line(img1, (int(x[index][0]), int(x[index][1])), (int(x[index][2]), int(x[index][3])),(0, 0, 255), 2, 16)

        print("--------------------------------")
        print("获取线的坐标为：")
        print(x[f])
        print(x[index])
        print("--------------------------------")

        print("确定船舶干舷位置：")
        middle_top_x = (x[f][0] + x[f][2])/2
        middle_top_y = (x[f][1] + x[f][3])/2
        print(middle_top_x, middle_top_y)

        k = (x[index][3] - x[index][1]) * 1.0 / (x[index][2] - x[index][0])
        a = k
        b = -1.0
        c = x[index][1] - k * x[index][0] #直线公式
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
        cv2.circle(img1, point, 5, (0, 0, 255), -1)
        cv2.circle(img1, point1, 8, (0, 255, 255), -1)
        cv2.circle(img1, point2, 8, (255, 255, 255), -1)


        end = time.time()

        print(end-start)



        cv2.imwrite("1.jpg", img1)


        print("-------------------------------------------------")



    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while (True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(deeplab.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

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
