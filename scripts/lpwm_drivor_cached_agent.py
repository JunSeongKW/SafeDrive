"""Official evaluation adapter: return sealed, sensor-only model predictions.

The scene is used solely for token lookup AFTER predictions have been generated;
no annotation or future value can be passed back to the model.
"""
from pathlib import Path
import numpy as np
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig, Trajectory


class CachedTrajectoryAgent(AbstractAgent):
    def __init__(self, trajectory_sampling, predictions_path):
        super().__init__(trajectory_sampling, requires_scene=True)
        self.predictions_path = Path(predictions_path)

    def name(self):
        return "LPWM_DrivoR_sealed_predictions"

    def get_sensor_config(self):
        return SensorConfig.build_no_sensors()

    def initialize(self):
        with np.load(self.predictions_path) as predictions:
            self.predictions = dict(zip(predictions['tokens'].tolist(), predictions['trajectories']))

    def compute_trajectory(self, agent_input, scene):
        return Trajectory(self.predictions[scene.scene_metadata.initial_token], self._trajectory_sampling)
