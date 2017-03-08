# -*- coding: utf-8 -*-
import cv2
import sys

dataset_dir = "images/"


def hog_func():
    # argv = sys.argv
    # argc = len(argv)
    # if (argc != 2):
    #     # 引数がちゃんとあるかチェック
    #     # 正しくなければメッセージを出力して終了
    #     print('Usage: python %s arg1' % argv[0])
    #     quit()
    # # 画像の読み込み
    # img_name = dataset_dir + argv[1]
    img_name = dataset_dir + 'liverpool_street.jpg'
    im = cv2.imread(img_name)
    cv2.imshow('defoult',im)
    # HoG特徴量の計算
    hog = cv2.HOGDescriptor()
    # SVMによる人検出
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    hogParams = {'winStride': (8, 8), 'padding': (32, 32), 'scale': 1.05}
    # 人を検出した座標
    human, r = hog.detectMultiScale(im, **hogParams)
    # 長方形で人を囲う
    for (x, y, w, h) in human:
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 50, 255), 3)
    # 人を検出した座標
    cv2.imshow("Human detection", im)
    cv2.waitKey(0)
    # 画像保存
    # out_file_name = "after" + argv[1]
    out_file_name = dataset_dir+"after" + 'Img3.bmp'
    cv2.imwrite(out_file_name, im)
    print("saved")


if __name__ == '__main__':
    hog_func()