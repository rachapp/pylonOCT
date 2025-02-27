import numpy as np
import h5py
from PyQt6.QtCore import QThread, pyqtSignal
from threading import Lock

# Lock to ensure thread-safe file access
file_lock = Lock()


class SaveDataThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool)

    def __init__(self, frame_buffer, output_file):
        super().__init__()
        self.frame_buffer = frame_buffer
        self.output_file = output_file
        self.is_running = False  # Add a flag to track thread status

    def run(self):
        if self.is_running:
            print("Save operation is already in progress. Skipping duplicate start.")
            return
        self.is_running = True
        try:
            total_frames = len(self.frame_buffer)
            chunk_size = 10
            frames_array = np.array(self.frame_buffer)
            num_frames = frames_array.shape[0]

            print(f"Starting to save {total_frames} frames to {self.output_file}")

            with h5py.File(self.output_file, "w") as hdf_file:
                dataset = hdf_file.create_dataset(
                    "frames",
                    shape=frames_array.shape,
                    dtype=frames_array.dtype,
                )

                for i in range(0, num_frames, chunk_size):
                    end = min(i + chunk_size, num_frames)
                    dataset[i:end] = frames_array[i:end]

                    progress_percent = int((end / num_frames) * 100)
                    print(f"Saving progress: {progress_percent}%")
                    self.progress.emit(progress_percent)

            print(f"Saved {total_frames} frames to {self.output_file}")
            self.finished.emit(True)
        except Exception as e:
            print(f"Error saving data: {e}")
            self.finished.emit(False)
        finally:
            self.is_running = False  # Reset flag when finished
