import time
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker, QWaitCondition

import numpy as np
from scipy.fft import rfft, next_fast_len
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator
from scipy.signal import get_window

class FFTThread(QThread):
    fft_ready = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)  # Signal to emit FFT result
    fps_signal = pyqtSignal(float)      # Signal to emit FPS

    def __init__(self, lambda_min=1200, lambda_max=1430, num_points=2048, peak_percentage=56, apply_lambda_to_k=True, apply_dc_subtraction=True):
        """
        Initialize the FFT thread.

        :param lambda_min: Minimum wavelength for interpolation.
        :param lambda_max: Maximum wavelength for interpolation.
        :param apply_lambda_to_k: Boolean to toggle lambda-to-k conversion.
        """
        super().__init__()
        self.frame = None
        self.fps = 0
        self.running = False
        # self.mutex = QMutex()
        # self.condition = QWaitCondition()

        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.num_points = num_points
        self.peak_percentage = peak_percentage
        self.apply_lambda_to_k = apply_lambda_to_k  # Toggle lambda-to-k conversion
        self.apply_dc_subtraction = apply_dc_subtraction

        self.update_lambda_arrays()
    
    def start(self):
        """Start the thread."""
        self.running = True
        super().start()

    def update_lambda_arrays(self):
        """
        Update the lambda and k arrays based on the current lambda_min and lambda_max.
        """
        self.lambda_array = np.linspace(self.lambda_min, self.lambda_max, self.num_points)
        self.k_array = 2 * np.pi / self.lambda_array[::-1]
        self.k_grid = np.linspace(np.min(self.k_array), np.max(self.k_array), self.num_points)

    def set_lambda_min(self, value):
        """
        Set a new minimum wavelength.
        """
        self.lambda_min = value
        self.update_lambda_arrays()

    def set_lambda_max(self, value):
        """
        Set a new maximum wavelength.
        """
        self.lambda_max = value
        self.update_lambda_arrays()

    def set_frame(self, frame):
        """
        Set the frame for FFT processing.
        """
        # with QMutexLocker(self.mutex):
        #     self.frame = frame
        #     self.condition.wakeOne()  # Notify the thread that a new frame is ready
        if self.running:
            # print("FFT")
            self.frame = frame
            # print("Frame set for FFT")

    def run(self):
        """
        Main loop for FFT computation.
        """
        self.running = True
        while self.running:
            if self.frame is None:
                time.sleep(0.000001)
                # self.msleep(1)  # Sleep briefly if no frame is available
                continue

            loop_start_time = time.perf_counter()

            # Apply dc_subtraction if enabled
            if self.apply_dc_subtraction:
                self.dc_subtraction()

            # Apply lambda_to_k if enabled
            if self.apply_lambda_to_k:
                self.lambda_to_k()

            # Calculate FFT
            self.calculate_fft()

            # Update FPS
            loop_time = time.perf_counter() - loop_start_time
            self.calculate_fps(loop_time)

            # Clear the frame after processing
            self.frame = None

    def dc_subtraction(self):
        """calculate dc and subtract the dc component"""
        if self.frame is None:
            return
        
        dc = np.mean(self.frame, axis=0)
        self.frame = self.frame - dc
    
    # this is faster lambda_to_k
    def lambda_to_k(self):
        """
        Convert the frame from lambda spacing to k spacing using optimized NumPy's interpolation.
        """
        if self.frame is None:
            return

        # Reverse the frame along the wavelength axis
        frame_reversed = self.frame[:, ::-1]

        # Perform fast linear interpolation using broadcasting
        self.frame = np.apply_along_axis(
            lambda row: np.interp(self.k_grid, self.k_array, row), axis=1, arr=frame_reversed
        )
    
    def calculate_fft(self):
        """
        Compute the FFT of the frame with symmetric zero-padding to 4096 samples.
        """
        if self.frame is None:
            return

        # Determine the target FFT size
        target_size = next_fast_len(2500)
        current_size = self.frame.shape[1]

        # Calculate the current peak position from the user-specified percentage
        current_peak_index = int(current_size * (self.peak_percentage / 100.0))

        # Calculate the desired peak position in the padded array (center of the target array)
        target_peak_index = target_size // 2

        # Compute the shift needed to move the peak to the center of the padded array
        shift = target_peak_index - current_peak_index

        # Calculate left and right padding based on the shift
        left_padding = max(0, shift)
        right_padding = max(0, target_size - current_size - left_padding)

        # Apply padding to the frame
        if target_size > current_size:
            # Apply asymmetric padding to center the peak
            frame_padded = np.pad(self.frame, ((0, 0), (left_padding, right_padding)), mode='constant')
        else:
            # No padding needed if already at the target size
            frame_padded = self.frame

        # Apply a window function (e.g., Hamming window)
        window = get_window("boxcar", current_size)  # Generate window for the current size
        window_padded = np.pad(window, (left_padding, right_padding), mode="constant")  # Correct 1D padding

        # Broadcast the window across rows of the frame
        frame_windowed = frame_padded * window_padded[np.newaxis, :]  # Apply the window

        # Compute FFT
        fft_result = rfft(frame_windowed, axis=1, overwrite_x=True)
        
        fft_result2 = np.where(fft_result < 400, 0, fft_result)
        # print(type(fft_result), type(fft_result2))
        # compute phase 
        # Compute the magnitude and exclude lower frequencies if necessary
        fft_magnitude = np.abs(fft_result[:, 25:525]).T  # Adjust frequency range if needed
        fft_phase = np.angle(fft_result2[0:998, 25:525]).T - np.angle(fft_result2[1:999, 25:525]).T

        # Emit the FFT result
        self.fft_ready.emit(fft_magnitude, frame_padded, fft_phase)

    def calculate_fps(self, loop_time):
        """
        Calculate frames per second (FPS).
        """
        self.fps = 1 / loop_time if loop_time > 0 else 0
        self.fps_signal.emit(self.fps)

    def stop(self):
        """
        Stop the FFT thread gracefully.
        """
        self.running = False
        self.wait()
