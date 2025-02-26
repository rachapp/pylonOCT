import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QTimer
from pypylon import pylon
# from pylongigetestcase import PylonTestCase
from pylonemutestcase import PylonTestCase
import time
from collections import deque


class CameraAcquisition(QThread):
    frame_ready = pyqtSignal(np.ndarray, int, float)
    camera_list_updated = pyqtSignal(list)

    def __init__(self, parent=None, use_emu=False):
        super().__init__(parent)
        self.is_running = False
        self.stack_time = deque(maxlen=10)
        self.acq_fps = 0
        self.frame_counter = 0
        self.camera = None
        self.converter = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_camera_connection)
        self.timer.start(1000)
        self.width = 0
        self.height = 0
        self.connected = False
        self.selected_camera_index = 0
        self.camera_list = []
        self.use_emu = use_emu
        if self.use_emu:
            self.pylon_emu = PylonTestCase()
        self.grabbing_active = False  # Add this flag
        self.update_camera_list()

    def run(self):
        self.is_running = True
        if not self.initialize_camera():
            return

        if self.camera is not None:
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self.grabbing_active = True  # Set the flag when grabbing starts

            while self.is_running:
                start_time = time.perf_counter()

                if not self.grabbing_active:  # Check the flag before RetrieveResult
                    break
                grabResult = None  # Initialize grabResult here
                try:
                    grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

                    if grabResult.GrabSucceeded():
                        frame = np.array(grabResult.Array)
                        self.frame_ready.emit(frame, self.frame_counter, self.acq_fps)
                        self.frame_counter += 1
                    else:
                        print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)

                except pylon.TimeoutException:
                    print("Timeout occurred while retrieving a grab result.")
                except pylon.GenericException as e:
                    print(f"Generic exception occurred: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
                finally:
                    if grabResult is not None:
                        try:
                            grabResult.Release()
                        except Exception as e:
                            print(f"Error during release of grab result: {e}")

                elapsed_time = time.perf_counter() - start_time
                self.stack_time.append(elapsed_time)
                self.acq_fps = round(1 / np.mean(self.stack_time), 2)

            self.camera.StopGrabbing()
            self.grabbing_active = False  # Reset the flag when grabbing stops

    def stop(self):
        self.is_running = False
        self.grabbing_active = False  # set the flag to false
        if self.camera is not None:
            try:
                if self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
                self.camera.Close()
                self.camera.DetachDevice()
            except Exception as e:
                print(f"Error during stop : {e}")

        self.quit()
        self.wait()

    def initialize_camera(self):
        try:
            if not self.camera_list:
                print("No cameras found.")
                self.connected = False
                return False

            selected_camera_info = self.camera_list[self.selected_camera_index]

            if self.use_emu:
                self.camera = self.pylon_emu.create_first()
            else:
                self.camera = pylon.InstantCamera(
                    pylon.TlFactory.GetInstance().CreateDevice(selected_camera_info)
                )

            if not self.use_emu:
                print("Using device ", self.camera.GetDeviceInfo().GetModelName())

            if self.camera is None:
                print("Failed to create camera instance.")
                self.connected = False
                return False

            self.camera.Open()
            self.width = self.camera.Width.Value = 4096
            self.height = self.camera.Height.Value = 1000

            self.camera.PixelFormat = "Mono10"
            self.camera.AcquisitionFrameRateEnable = True
            self.camera.AcquisitionFrameRate.SetValue(50)

            self.camera.MaxNumBuffer = 5
            self.connected = True
            return True

        except Exception as e:
            print(f"Error initializing camera: {e}")
            self.connected = False
            return False

    def check_camera_connection(self):
        try:
            if not self.connected:
                self.initialize_camera()
            else:
                if not self.camera.IsCameraDeviceRemoved():
                    pass
                else:
                    self.connected = False
                    self.camera.DetachDevice()
                    self.camera.Close()
                    self.camera = None
                    print("Camera is disconnected!")
                    self.update_camera_list()  # Update camera list
        except:
            pass

    def update_camera_list(self):
        try:
            tl_factory = pylon.TlFactory.GetInstance()

            if self.use_emu:
                devices = tl_factory.EnumerateDevices(self.pylon_emu.device_filter)
            else:
                devices = tl_factory.EnumerateDevices()

            self.camera_list = devices
            camera_names = []
            if not devices:
                print("No camera present.")
            else:
                for i, device_info in enumerate(devices):
                    camera_names.append(f"{device_info.GetModelName()} ({device_info.GetSerialNumber()})")

            self.camera_list_updated.emit(camera_names)
        except Exception as e:
            print(f"Error updating camera list: {e}")

    def set_selected_camera(self, index):
        if 0 <= index < len(self.camera_list):
            if self.camera is not None:
                self.stop()
                self.camera = None
            self.selected_camera_index = index
            self.connected = False  # Force a reconnection with the new camera
            print(f"Selected camera index: {self.selected_camera_index}")
        else:
            print("Invalid camera index selected.")
