import sys

from PyQt5 import QtWidgets

from sketch.WindowUI import WindowUI

from options import TestOptions

if __name__ == "__main__":
    opt = TestOptions().parse()
    # dataset = data.create_dataset(opt)
    app = QtWidgets.QApplication([])
    ui = WindowUI(opt)
    ui.show()
    sys.exit(app.exec_())
