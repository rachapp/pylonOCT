import sys
from PyQt6 import QtWidgets, QtCore
from basic_grab_gui import Ui_MainWindow
from camera_acquisition_dummy import CameraAcquisition
import numpy as np
from numpy.fft import rfft, rfftfreq


class MainApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        self.setupUi(self)

        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(390, 240, 150, 31))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("Default camera")
        self.comboBox.currentIndexChanged.connect(self.on_camera_selection_changed)

        self.camera_thread = CameraAcquisition(use_emu=False) # change to false when you want to connect with a real camera
        self.camera_thread.camera_list_updated.connect(self.update_camera_list)
        self.camera_thread.frame_ready.connect(self.update_image)
        self.is_running = False
        self.camera_thread.finished.connect(self.on_camera_thread_finished)

        self.pushButton.clicked.connect(self.start_acquisition)
        self.pushButton_2.clicked.connect(self.stop_acquisition)

        self.plot_update_interval = 5  # Update plots every 5 frames

    def start_acquisition(self):
        if not self.is_running:
            print("Starting acquisition...")
            try:
                if self.camera_thread.camera is None:
                    if not self.camera_thread.initialize_camera():
                        print("Failed to initialize camera.")
                        return

                self.camera_thread.start()
                self.is_running = True
                self.pushButton_2.setEnabled(True)
                self.pushButton.setText("Running")
                self.pushButton.setEnabled(False)
            except Exception as e:
                print(f"Error starting acquisition: {e}")

    def stop_acquisition(self):
        if self.is_running:
            print("Stopping acquisition...")
            self.camera_thread.stop()
            self.is_running = False
            self.pushButton.setText("Start")
            self.pushButton.setEnabled(True)
            self.pushButton_2.setEnabled(False)
            print("Acquisition stopped.")

    def on_camera_thread_finished(self):
        """Called when the camera thread has finished."""
        print("Camera thread finished.")
        self.camera_thread.frame_counter = 0
        self.camera_thread.camera = None

    def update_image(self, frame, frame_number, acq_fps):
        self.widget.setImage(frame.T, autoLevels=False, levels=(0, 1023))
        self.label.setText(f"Frame No: {frame_number}")
        self.label_4.setText(f"FPS: {acq_fps}")

        # Update plots only every N frames
        if frame_number % self.plot_update_interval == 0:
            self.update_plot(frame)

    def update_plot(self, frame):
        self.widget_2.clear()
        self.widget_3.clear()

        line_profile = frame[int(frame.shape[0] / 2), :]
        x = np.linspace(2 * np.pi / 0.9, 2 * np.pi / 0.7, frame.shape[1])
        self.widget_2.plot(x, line_profile, clear=True)

        # Compute the FFT of the line profile
        fft_line_profile = np.abs(rfft(line_profile))[10:500]  # Exclude DC component
        fft_x = rfftfreq(len(line_profile))[10:500]  # Exclude DC component
        self.widget_3.plot(fft_x, fft_line_profile, clear=True)

    def update_camera_list(self, camera_names):
        self.comboBox.clear()
        self.comboBox.addItems(camera_names)
        self.comboBox.addItem("Dummy camera")

    def on_camera_selection_changed(self, index):
        print("camera changed")
        if self.camera_thread.camera is not None:
            self.camera_thread.set_selected_camera(index)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainApp()
    main_window.show()
    main_window.pushButton_2.setEnabled(False)
    sys.exit(app.exec())
