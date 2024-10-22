import os

from src.helpers import get_sim
from src.simulation import *
import torch
from src.config import Config
from tqdm import tqdm

def gen_data(config: Config):
    torch.manual_seed(config.seed)

    # Check if we actually need to generate data.
    already_generated = (len(os.listdir(config.get_data_path())) >=
                         config.number_of_examples)
    if already_generated or config.gen_during_training:
        return

    out = config.get_data_path()
    out.mkdir(exist_ok=True)

    # Create matrix if needed.
    if config.visualize_data_gen:
        matrix = LEDMatrix(config.matrix_width,
                           config.matrix_length,
                           config.pallete_size)
    else:
        matrix = None

    # Create and run simulation.
    sim = get_sim(config, matrix=matrix)

    for _ in tqdm(range(config.number_of_examples)):
        for _ in range(config.model_interval):
            sim.advance(config.simulation_interval)
            if config.visualize_data_gen:
                time.sleep(1 / config.simulation_fps)
        sim.save_board()