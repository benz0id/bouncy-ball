import time

import torch
from pathlib import Path
from src.simulation import LEDMatrix, Simulation
from typing import List
from collections import deque

from src.config import Config
from src.helpers import pred_to_frame, pred_to_heatmap, FrameCache, get_sim
from copy import deepcopy


def evaluate(config: Config):
    torch.manual_seed(config.seed)

    # Create frame buffer (for model input) using either a fresh simulation
    # or some training data.
    if config.new_simulation:
        sim = get_sim(config)
        initial = []
        for _ in range(config.number_of_prev_frames):
            for _ in range(config.model_interval):
                sim.advance(config.simulation_interval)
            initial.append(deepcopy(sim.board).to(config.device))
    else:
        initial = [
            torch.load(config.get_data_path() / f'{i}.pt').to(config.device).to(torch.float32)
            for i in range(config.number_of_prev_frames)]
    cache = FrameCache(initial)

    matrix = LEDMatrix(
        config.matrix_width,
        config.matrix_length,
        config.pallete_size
    )

    model = torch.load(config.get_model_path()).to(config.device).eval()

    # Counter for current frame in training data.
    i = config.number_of_prev_frames

    if config.new_simulation and config.show_actual:
        matrix.set_layer('actual', sim.board)

    while True:
        # Use the model to predict the next state.
        last_states = cache.get_nn_input().to(torch.float32).to(config.device)
        next_state_pred = model(last_states)

        # Update cache using the predicted state.
        next_state = pred_to_frame(next_state_pred.unsqueeze(0)).squeeze()
        cache.update(next_state.to(config.device))

        # Visualise softmaxed NN output as heatmap.
        heatmap = pred_to_heatmap(next_state_pred,
                                  config.heatmap_palette_size,
                                  config.heatmap_palette_offset)

        if config.new_simulation and config.show_actual:
            for _ in range(config.model_interval):
                sim.advance(config.simulation_interval)
        elif config.show_actual:
            actual = torch.load(config.get_data_path() / f'{i}.pt')
            matrix.set_layer('actual', actual)
            i += 1

        if config.use_heatmap:
            matrix.set_layer('heatmap', heatmap)
        else:
            # Use last color on heatmap for simulation ball.
            matrix.set_layer('pred', next_state +
                             config.heatmap_palette_offset +
                             config.heatmap_palette_size - 1)
        time.sleep(1 / config.simulation_fps)
