"""ATC simulation environment.

Top-level entry points for using the simulation as an RL environment:

    from environment import SimulationEnv, HumanDataRecorder

    sim = SimulationEnv(airport_name='test', star_mode=True, spawn_single=True)
    state = sim.get_state()
    sim.command(callsign, 'C 270')
    sim.step(1.0)

See doc/ for architecture, behavior, and logger references.
"""

from environment.core.simulation import SimulationEnv
from environment.core.human_data_logger import HumanDataRecorder
from environment.core.aircraft import Aircraft

__all__ = ['SimulationEnv', 'HumanDataRecorder', 'Aircraft']
