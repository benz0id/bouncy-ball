import dataclasses
from pathlib import Path
from typing import List, Tuple, Union

import torch


@dataclasses.dataclass
class Config:
    run_name: str
    data_storage_dir: Path
    device: str

    simulation_fps: int
    visualize_data_gen: bool
    visualize_training: bool
    skip_epochs: List[int]
    pallete_size: int
    heatmap_palette_size: int
    heatmap_palette_offset: int
    matrix_width: int
    matrix_length: int

    model_interval: int
    number_of_examples: int

    seed: int
    simulation_interval: float
    starting_pos: Union[Tuple[int, int], None]
    vertical_acceleration: float
    lateral_acceleration: float
    number_of_balls: int
    ball_lat_potential: float
    ball_vert_potential: float
    randomize_potential: bool

    model: torch.nn.Module
    number_of_prev_frames: int
    epochs: int
    batch_size: int

    gen_during_training: bool
    num_simulation_threads: int
    model_caching: bool
    optimizer: torch.optim.Optimizer
    scheduler: Union[torch.optim.lr_scheduler.LRScheduler, None]
    loss_fxn: torch.nn.Module
    use_heatmap: bool
    new_simulation: bool
    show_actual: bool

    def get_data_path(self) -> Path:
        return self.data_storage_dir / self.run_name / 'training_data'

    def get_model_path(self) -> Path:
        return self.data_storage_dir / self.run_name / 'model.pt'

    def get_cached_model_path(self) -> Path:
        return self.data_storage_dir / self.run_name / 'cached_model.pt'

    def get_config_path(self) -> Path:
        return self.data_storage_dir / self.run_name / 'config'
