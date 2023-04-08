from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#多线程
from multiprocessing import Process
import time,threading

from PyQt5.QtCore import QThread
from PyQt5.Qt import (QApplication, QWidget, QPushButton, QThread, QMutex, pyqtSignal)

#GUI界面
from PyQt5 import QtGui,QtCore
from PyQt5.QtWidgets import *
from Designer.Loading import Ui_LoadingWindow
from Designer.Main import Ui_MainWindow
import sys
import shutil

#faster-rcnn
import argparse
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer




CLASSES = ('__background__', "bullet", "dagger", "defibrillator", "fruit_knife", "gas", "lighter_gas", "national_knife","paper_knife", "simulation_gun", "switchblade")

NETS = {'vgg16': ('vgg16.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval',)}


#中文转化
def translate(class_name):
    final_class_name = "未知"
    if class_name == "bullet":
        final_class_name = "子弹"
    elif class_name == "dagger":
        final_class_name = "匕首"
    elif class_name == "defibrillator":
        final_class_name = "电击器"
    elif class_name == "fruit_knife":
        final_class_name = "水果刀"
    elif class_name == "gas":
        final_class_name = "瓦斯"
    elif class_name == "lighter_gas":
        final_class_name = "打火机气"
    elif class_name == "national_knife":
        final_class_name = "民族刀"
    elif class_name == "paper_knife":
        final_class_name = "裁剪刀"
    elif class_name == "simulation_gun":
        final_class_name = "仿真枪"
    elif class_name == "switchblade":
        final_class_name = "弹簧刀"
    else:
        final_class_name = "未知"
    return final_class_name

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]#准确率

        # 画出边框
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )

        # 中文转化
        final_class_name = translate(class_name)

    plt.rcParams['font.sans-serif'] = ['SimHei']

    plt.rcParams['axes.unicode_minus'] = False

    #标记类别
    ax.text(bbox[0], bbox[1] - 2,
            final_class_name,
            bbox=dict(facecolor='blue', alpha=0.5),
            fontsize=40, color='white')

    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.savefig('output/images/{:}.jpg'.format(score))#将图片以准确率为名进行保存


def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""
    # Load the demo image
    im_file = os.path.join(cfg.FLAGS2["data_dir"], 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    # print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.6
    NMS_THRESH = 0.1
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]


        vis_detections(im, cls, dets, thresh=CONF_THRESH)

    Bin.shibie_bool = True


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args

#初始化模型
def init_model():
    # time.sleep(10)
    # print(os.getpid(),"init_____________________________________")
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default', NETS[demonet][0])
    print(tfmodel,"xxx")

    if not os.path.isfile(tfmodel + '.meta'):
        print(tfmodel)
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = False

    # init session
    sess = tf.Session(config=tfconfig)
    Bin.sess = sess

    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
        Bin.net = net
    else:
        raise NotImplementedError

    n_classes = len(CLASSES)
    # create the structure of the net having a certain shape (which depends on the number of classes)
    net.create_architecture(sess, "TEST", n_classes,
                            tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)


    print('Loaded network {:s}'.format(tfmodel))
    Bin.init_model_bool = True





#界面与模型的交互类
class Bin:
    image_url = None#需要识别图片的地址
    image_name = None

    sess = None
    net = None

    init_model_bool = False

    shibie_bool = False

    filepath = None


    final_label = None


class LoadWindow(QMainWindow,Ui_LoadingWindow):
    def __init__(self,*args,**kwargs):
        super(LoadWindow, self).__init__(*args,**kwargs)
        self.setupUi(self)
        self.pushButton.setShortcut(QtCore.Qt.Key_Return)
        self.pushButton.clicked.connect(self.bth_login_fuc)



    def bth_login_fuc(self):
        user = self.lineEdit.text()
        password = self.lineEdit_2.text()

        if user=="admin" and password=="admin":
            self.main_window = MainWindow()
            self.main_window.show()
            self.close()

        else:
            QMessageBox.information(None,'登录错误',"账号或密码错误")

qmut_1 = QMutex()  # 创建线程锁
class WorkThread1(QThread):
    init_signal = pyqtSignal()
    def __int__(self):
        super(WorkThread1, self).__init__()

    def run(self):
        qmut_1.lock()  # 加锁

        while True:
            time.sleep(0.2)#每0.2s监听一次模型是否完成
            if Bin.init_model_bool==False:
                pass
            else:
                self.init_signal.emit()
                break
        qmut_1.unlock()#解锁

qmut_2 = QMutex()  # 创建线程锁
class WorkThread(QThread):
    display = pyqtSignal()
    hidden = pyqtSignal()
    demo = pyqtSignal()

    show_image = pyqtSignal()
    show_empty = pyqtSignal()

    def __int__(self):
        super(WorkThread, self).__init__()

    def show_class_image(self):
        if os.path.exists(Bin.filepath):
            # 找到文件夹中最大的值，显示在qt中
            filelist = os.listdir("output/images")
            if filelist:
                for i in range(len(filelist)):
                    filelist[i] = float(filelist[i][:-4])

                # 找到准确率(文件名)最大的进行图片显示
                filelist.reverse()
                final_label = filelist[0]
                Bin.final_label = final_label

                #显示图片
                self.show_image.emit()
            else:
                #显示空包
                self.show_empty.emit()

    def run(self):
        qmut_2.lock()  # 加锁
        self.display.emit()
        # print("start")
        demo(Bin.sess, Bin.net, Bin.image_url)
        self.show_class_image()
        # print("end")
        self.hidden.emit()
        qmut_2.unlock()#解锁

class MainWindow(QMainWindow,Ui_MainWindow):

    def __init__(self,*args,**kwargs):
        super(MainWindow, self).__init__(*args,**kwargs)
        self.setupUi(self)

        self.pushButton.setEnabled(False)
        self.pushButton.setStyleSheet("background-color: rgb(211, 211, 211);\n"
"color: rgb(0, 0, 0);\n"
"border:0px;\n"
"border-radius:10px;")
        self.pushButton.setText("导入模型中")
        self.pushButton.setShortcut(QtCore.Qt.Key_Return)
        self.pushButton.clicked.connect(self.openfile)

        #等待gif
        self.gif = QtGui.QMovie('output/res/process2.gif')
        self.label_5.setMovie(self.gif)
        self.gif.start()
        self.label_5.setVisible(False)
        self.label_6.setVisible(False)

        self.workThread1 = WorkThread1()
        self.workThread1.start()
        self.workThread1.init_signal.connect(self.init_model)


    def init_model(self):
        # 监听t2数据是否导入完成，完成则可用
        self.pushButton.setEnabled(True)
        self.pushButton.setStyleSheet("background-color: rgb(85, 170, 255);\n"
                                      "color: rgb(255, 255, 255);\n"
                                      "border:0px;\n"
                                      "border-radius:10px;")
        self.pushButton.setText("请导入图片")
        # print("模型导入成功")

    def display(self):
        # print("dispaly")
        img = QtGui.QPixmap(Bin.image_url).scaled(self.label_4.width(), self.label_4.height())
        self.label_4.setPixmap(img)

        self.label_5.setVisible(True)
        self.label_6.setVisible(True)
    def hidden(self):
        # print("hidden")
        self.label_5.setVisible(False)
        self.label_6.setVisible(False)

    def show_image(self):
        self.label_3.setVisible(True)
        label_image = QtGui.QPixmap("output/images/" + str(Bin.final_label) + ".jpg").scaled(self.label_3.width(),
                                                                                         self.label_3.height())
        self.label_3.setPixmap(label_image)
    def show_empty(self):
        self.label_3.setVisible(True)
        self.label_3.setText("空包")

    def openfile(self):

        #清空文件夹
        filepath = 'output/images'
        Bin.filepath = filepath
        shutil.rmtree(filepath)
        os.mkdir(filepath)

        imgUrl, imgType = QFileDialog.getOpenFileName(self,'请选择图片','demo','Image files(*.jpg);;All Files(*)')
        Bin.image_name = imgUrl.split("/")[::-1][0]
        Bin.image_url = imgUrl


        # self.workThread.wait()
        if Bin.image_url and True:
            self.label_3.setVisible(False)
            self.workThread = WorkThread()
            self.workThread.start()
            self.workThread.display.connect(self.display)

            self.workThread.show_image.connect(self.show_image)
            self.workThread.show_empty.connect(self.show_empty)

            self.workThread.hidden.connect(self.hidden)
        else:
            self.label_3.setVisible(True)
            # self.label_3.setText("未输入图片")

def gui():
    app = QApplication(sys.argv)
    load = LoadWindow()
    load.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    t1 = threading.Thread(target=gui)
    t1.start()
    t2 = threading.Thread(target=init_model)
    t2.start()
    t1.join()

