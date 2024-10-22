from collections import deque
from typing import List

import torch

from src.config import Config
from src.simulation import Simulation, LEDMatrix


def pred_to_frame(pred: torch.tensor, reshape: bool = True,
                  n_balls: int = 1) -> torch.tensor:
    """
    Converts raw NN output into frames in the same format as those in the
    simulation.
    """
    greatest_probs = pred.argsort(-1, descending=True)
    frame = torch.zeros_like(pred, dtype=torch.bool)
    for i in range(len(frame)):
        frame[i][greatest_probs[i][:n_balls]] = True
    if reshape:
        frame = frame.view(-1, 64, 32)
    return frame


def pred_to_heatmap(
        pred: torch.tensor,
        heatmap_pal_size: int,
        heatmap_pal_offset: int,
        reshape: bool = True
    ) -> torch.tensor:
    """
    Applies transformations to raw  to create a visually appealing heatmap.
    Colours denoted as indices in a gradient palette.

    :param pred: Raw NN output.
    :param heatmap_pal_size: The number of colours available to the heatmap.
    :param heatmap_pal_offset: Right shift pallete by this amount.
    :param reshape: Whether to reshape to matric size.
    :return: Array of palette values.
    """
    probs = torch.softmax(pred, 0)
    probs /= probs.max().item()
    probs = probs ** (3 / 4)
    grad = (probs * heatmap_pal_size).to(torch.int32) + heatmap_pal_offset
    if reshape:
        grad = grad.view(64, 32)
    return grad


def get_sim(config: Config, device: str = None, matrix: LEDMatrix = None):
    """
    Create a simulation satifying the criteria outlined in <config>
    """
    gravity = torch.tensor((
        config.lateral_acceleration,
        config.vertical_acceleration
    ),
        dtype=torch.float32)
    sim = Simulation(config.get_data_path(), matrix, gravity, device=device)
    sim.add_random_ball_potential(
        n=config.number_of_balls,
        lat_potential=config.ball_lat_potential,
        vert_potential=config.ball_vert_potential,
        randomize_potential=config.randomize_potential
    )
    return sim


class FrameCache:
    """
    Keeps a queue of previous frames from a simulation. Provides frames
    for use as input to neaural networks.
    """
    stack: deque

    def __init__(self, frames: List[torch.tensor]):
        self.stack = deque(frames)

    def update(self, frame: torch.tensor):
        self.stack.popleft()
        self.stack.append(frame)

    def get_nn_input(self) -> torch.tensor:
        return torch.cat(list(self.stack), 0).flatten()
