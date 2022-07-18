import os
import time
from time import gmtime, strftime

import cv2
from PIL.Image import Image
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QGraphicsView

from util.memcache import Memcached
from .Input_mouse_event import InputGraphicsScene
from .Output_mouse_event import OutputGraphicsScene
from .SketchGUI import Ui_SketchGUI
from .model_mse import SketchGenerator


class WindowUI(QtWidgets.QMainWindow, Ui_SketchGUI):

    def __init__(self, opt):
        super(WindowUI, self).__init__()
        self.setupUi(self)
        self.setEvents()
        self._translate = QtCore.QCoreApplication.translate

        self.output_img = None
        self.brush_size = self.BrushSize.value()
        self.eraser_size = self.EraseSize.value()

        self.modes = [0, 1, 0]  # 0 marks the eraser, 1 marks the brush
        self.Modify_modes = [0, 1, 0]  # 0 marks the eraser, 1 marks the brush

        self.output_scene = OutputGraphicsScene(self)
        self.output.setScene(self.output_scene)
        self.output.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.output.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.output.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.output_view = QGraphicsView(self.output_scene)
        # self.output_view.fitInView(self.output_scene.updatePixmap())

        self.sketch_gen = SketchGenerator('checkpoints/model_mse.pth')

        self.input_scene = InputGraphicsScene(self.modes, self.brush_size, self.output_scene, self.sketch_gen, opt,
                                              self)
        self.input.setScene(self.input_scene)
        self.input.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.input.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.input.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.input_scene.convert_on = self.RealTime_checkBox.isChecked()
        self.output_scene.convert_on = self.RealTime_checkBox.isChecked()

        self.BrushNum_label.setText(self._translate("SketchGUI", str(self.brush_size)))
        self.EraserNum_label.setText(self._translate("SketchGUI", str(self.eraser_size)))

        self.start_time = time.time()
        # self.

        # try:
        #     # thread.start_new_thread(self.output_scene.fresh_board,())
        #     thread.start_new_thread(self.input_scene.thread_shadow,())
        # except:
        #     print("Error: unable to start thread")
        # print("Finish")

    def setEvents(self):
        self.Undo_Button.clicked.connect(self.undo)

        self.Brush_Button.clicked.connect(self.brush_mode)
        self.BrushSize.valueChanged.connect(self.brush_change)

        self.Clear_Button.clicked.connect(self.clear)

        self.Eraser_Button.clicked.connect(self.eraser_mode)
        self.EraseSize.valueChanged.connect(self.eraser_change)

        self.Save_Button.clicked.connect(self.saveFile)

        self.Load_Button.clicked.connect(self.open)

        self.Convert_Sketch.clicked.connect(self.convert)
        self.RealTime_checkBox.clicked.connect(self.convert_on)

        self.actionSave.triggered.connect(self.saveFile)

    def mode_select(self, mode):
        for i in range(len(self.modes)):
            self.modes[i] = 0
        self.modes[mode] = 1

    def brush_mode(self):
        self.mode_select(1)
        self.brush_change()
        self.statusBar().showMessage("Brush")

    def eraser_mode(self):
        self.mode_select(0)
        self.eraser_change()
        self.statusBar().showMessage("Eraser")

    def undo(self):
        self.input_scene.undo()
        self.output_scene.undo()

    def brush_change(self):
        self.brush_size = self.BrushSize.value()
        self.BrushNum_label.setText(self._translate("SketchGUI", str(self.brush_size)))
        if self.modes[1]:
            self.input_scene.paint_size = self.brush_size
            self.input_scene.paint_color = (0, 0, 0)
        self.statusBar().showMessage("Change Brush Size in ", self.brush_size)

    def eraser_change(self):
        self.eraser_size = self.EraseSize.value()
        self.EraserNum_label.setText(self._translate("SketchGUI", str(self.eraser_size)))
        if self.modes[0]:
            print(self.eraser_size)
            self.input_scene.paint_size = self.eraser_size
            self.input_scene.paint_color = (1, 1, 1)
        self.statusBar().showMessage("Change Eraser Size in ", self.eraser_size)

    def clear(self):
        self.input_scene.reset()
        self.output_scene.reset()
        self.start_time = time.time()
        self.statusBar().showMessage("Clear Drawing Board")

    def convert(self):
        self.statusBar().showMessage("Press Convert")
        self.input_scene.generate_dpmap()
        self.input_scene.convert_RGB()
        self.output_scene.updatePixmap()

    def open(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                                                  QDir.currentPath(), "Images Files (*.*)")  # jpg;*.jpeg;*.png
        if fileName:  # the name of the dpmap
            dir_name, base_name = os.path.split(fileName)
            # load input file, data transform and set both seg and dpmap
            image = QPixmap(fileName)
            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                                        "Cannot load %s." % fileName)
                return
            dpmap_img = Memcached.cv2_imread(fileName)
            seg_img = self.sketch_gen.dpmap2seg(dpmap_img)

            self.input_scene.setSketchImag(dpmap_img, seg_img, fileName)

    def saveFile(self):
        cur_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        file_dir = './saveImage/' + cur_time
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir)

        cv2.imwrite(file_dir + '/hand-draw.jpg', self.input_scene.sketch_img * 255)
        cv2.imwrite(file_dir + '/colorized.jpg', cv2.cvtColor(self.output_scene.ori_img, cv2.COLOR_BGR2RGB))

        print(file_dir)

    def convert_on(self):
        # if self.RealTime_checkBox.isCheched():
        print('self.RealTime_checkBox', self.input_scene.convert_on)
        self.input_scene.convert_on = self.RealTime_checkBox.isChecked()
        self.output_scene.convert_on = self.RealTime_checkBox.isChecked()
