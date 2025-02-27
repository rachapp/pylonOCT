import nidaqmx
from nidaqmx.constants import AcquisitionType, Level, Edge
import numpy as np

def generate_raster_scan_wave_2d(n_steps_x, n_steps_y, X_amplitude, Y_amplitude, number_of_samples):
    """Generates a 2D raster scan pattern with voltage steps for both X and Y axes."""
    # Create X and Y step sequences
    x_values = np.linspace(-X_amplitude, X_amplitude, n_steps_x)
    y_values = np.linspace(-Y_amplitude, Y_amplitude, n_steps_y)
    
    # Create a 2D grid of X and Y values (raster scan pattern)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    
    # Flatten the grid to create a sequence of X and Y values
    x_wave = x_grid.flatten()
    y_wave = y_grid.flatten()
    
    # Repeat the values across the samples if necessary to match number_of_samples
    x_wave_repeated = np.tile(x_wave, number_of_samples // len(x_wave))[:number_of_samples]
    y_wave_repeated = np.tile(y_wave, number_of_samples // len(y_wave))[:number_of_samples]
    
    return x_wave_repeated, y_wave_repeated

def main():
    try:
        with nidaqmx.Task() as co_task:
            # Configure Counter Output for pulse train
            channel = co_task.co_channels.add_co_pulse_chan_freq(
                "Dev1/ctr0",
                idle_state=Level.LOW,
                initial_delay=0.0,
                freq=10.0,
                duty_cycle=0.1,
            )
            # Set the output terminal for the pulse train
            channel.co_pulse_term = "/Dev1/PFI5"
            co_task.timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS)

            with nidaqmx.Task() as ao_task:
                sampling_rate = 2000.0
                number_of_samples = 2000
                
                # Add two analog output channels: ao0 for triangle wave, ao1 for step wave
                ao_task.ao_channels.add_ao_voltage_chan("Dev1/ao0")  # Triangle wave
                ao_task.ao_channels.add_ao_voltage_chan("Dev1/ao1")  # Step wave
                
                # Configure Analog Output timing
                ao_task.timing.cfg_samp_clk_timing(
                    sampling_rate, sample_mode=AcquisitionType.CONTINUOUS
                )

                # Configure the start trigger for Analog Output
                ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                    trigger_source="/Dev1/PFI0", trigger_edge=Edge.RISING
                )

                actual_sampling_rate = ao_task.timing.samp_clk_rate
                print(f"Actual sampling rate: {actual_sampling_rate:g} S/s")
                
                # Generate the raster scan pattern
                x_wave, y_wave = generate_raster_scan_wave_2d(
                n_steps_x=100, n_steps_y=20, X_amplitude=1.0, Y_amplitude=1.0, number_of_samples=number_of_samples
                )

                # Stack the X and Y waveforms together for simultaneous output
                waveform_data = np.vstack([x_wave, y_wave])

                # Write and start the tasks
                ao_task.write(waveform_data, auto_start=False)
                ao_task.start()
                co_task.start()
                
                input("Generating pulse train. Press Enter to stop.\n")
                
                co_task.stop()
                ao_task.stop()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

