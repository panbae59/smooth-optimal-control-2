import numpy as np
from numpy.lib.npyio import save
from optimal_control.optimal_control import OptimalControl
from util.utils import save_value, load_value, get_closest_multiple_of_16
from qiskit.pulse import library as pulse_lib
from qiskit.pulse import Play
from util.sweep import Sweep

class GaussianControl(OptimalControl):
    def __init__(self, Q_setup):
        super().__init__()
        self.Q_setup = Q_setup
        self.set_pulse_params()
        self.set_gates()

    def set_pulse_params(self,):
        self.drive_sigma_us = 0.075
        self.drive_samples_us = self.drive_sigma_us*8

        self.drive_sigma = get_closest_multiple_of_16(self.drive_sigma_us * self.Q_setup.us / self.Q_setup.dt)       # The width of the gaussian in units of dt
        self.drive_samples = get_closest_multiple_of_16(self.drive_samples_us * self.Q_setup.us / self.Q_setup.dt)   # The truncating 


    def get_raw_gaussian_Signal(self, ):
        return np.array(pulse_lib.gaussian(duration=self.drive_samples, amp=1, 
                                        sigma=self.drive_sigma).samples)

    def set_gates(self, ):
        pi_amp = load_value("pi_amp")
        pi_pulse = pulse_lib.gaussian(duration=self.drive_samples,
                                    amp=pi_amp, 
                                    sigma=self.drive_sigma,
                                    name='pi_pulse')

        halfpi_pulse = pulse_lib.gaussian(duration=self.drive_samples,
                              amp=pi_amp/2, 
                              sigma=self.drive_sigma,
                              name='halfpi_pulse')

        y_pi_pulse = pulse_lib.gaussian(duration=self.drive_samples,
                                    amp=pi_amp *1j, 
                                    sigma=self.drive_sigma,
                                    name='y_pi_pulse')

        y_halfpi_pulse = pulse_lib.gaussian(duration=self.drive_samples,
                                    amp=pi_amp/2*1j, 
                                    sigma=self.drive_sigma,
                                    name='y_halfpi_pulse')

        self.gate['X']   = Play(pi_pulse, self.Q_setup.drive_chan)
        self.gate['X/2'] = Play(halfpi_pulse, self.Q_setup.drive_chan)
        self.gate['-X/2'] = Play(pulse_lib.Waveform(-halfpi_pulse.samples), self.Q_setup.drive_chan)
        self.gate['Y']   = Play(y_pi_pulse, self.Q_setup.drive_chan)
        self.gate['Y/2'] = Play(y_halfpi_pulse, self.Q_setup.drive_chan)
        self.gate['-Y/2'] = Play(pulse_lib.Waveform(-y_halfpi_pulse.samples), self.Q_setup.drive_chan)


    def tune_gaussain_pulse(self, ):
        from util.utils import baseline_remove, fit_function

        sweep = Sweep(self.Q_setup)

        num_rabi_points = 50

        drive_amp_min = 0
        drive_amp_max = 0.35
        drive_amps = np.linspace(drive_amp_min, drive_amp_max, num_rabi_points)

        self.set_pulse_params()

        rabi_signal = self.get_raw_gaussian_Signal()

        rabi_results = sweep.amplitude_sweep(drive_amps, [rabi_signal])

        import matplotlib.pyplot as plt
        rabi_values = []
        #print(rabi_results)
        for i in range(num_rabi_points):
            # Get the results for `qubit` from the ith experiment
            rabi_values.append(rabi_results[i]['0'])

        rabi_values = np.real(baseline_remove(rabi_values))

        fit_params, y_fit = fit_function(drive_amps,
                                 rabi_values, 
                                 lambda x, A, B, drive_period, phi: (A*np.cos(2*np.pi*x/drive_period - phi) + B),
                                 [400, 10, 0.3, 0])

        plt.scatter(drive_amps, rabi_values, color='black')
        plt.plot(drive_amps, y_fit, color='red')

        drive_period = fit_params[2] # get period of rabi oscillation
        plt.xlabel("Drive amp [a.u.]", fontsize=15)
        plt.ylabel("Measured signal [a.u.]", fontsize=15)
        plt.savefig("optimal_control\\gaussian_rabi_oscillate")

        pi_amp = abs(drive_period / 2)
        save_value("pi_amp" ,pi_amp)
        print(f"Pi Amplitude = {pi_amp}")

        # for our SOC pulses
        convert_factor = pi_amp*0.626617/(4/np.pi)/(4*1e6)
        save_value('convert_factor', convert_factor)


