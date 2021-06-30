from optimal_control.gaussian_control import GaussianControl
from optimal_control.paper_smooth_optimal_control import PaperSmoothOptimalControl
from optimal_control.qiskit_control import QiskitControl
from util.Q_setup import QSetUp

Q_setup = QSetUp()

# GOC = GaussianControl(Q_setup)
# GOC.tune_gaussain_pulse()

QC = QiskitControl(Q_setup)

PSOC = PaperSmoothOptimalControl(Q_setup)
PSOC.tune_paper_SOC_pulse(QC.gate['X/2'])
PSOC.set_gates()