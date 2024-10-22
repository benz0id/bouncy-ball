import datetime
from copy import deepcopy

from src.simulation import *
from pathlib import Path
import torch
from src.config import Config
from src.helpers import pred_to_heatmap, FrameCache, get_sim, pred_to_frame
from tqdm import tqdm
from multiprocessing import Queue
import multiprocessing as mp

class Process(mp.Process):
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            mp.Process.run(self)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            raise e  # You can still rise this exception if you need to

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


class StoredBallDataset(torch.utils.data.Dataset):
    """
    Loads pre-computed simulation data for use in training.
    """
    dir_path: Path

    length: int
    config: Config
    device: str

    def __init__(self,
                 config: Config,
                 device: str,
                 dir_path: Path,
                 length: int):

        self.dir_path = dir_path
        self.length = length
        self.config = config
        self.device = device

    def __len__(self):
        return self.length - self.config.number_of_prev_frames

    def get_sim_state(self, i: int):
        return torch.load(self.dir_path / f'{i}.pt').to(torch.float32).to(
            self.device)

    def __getitem__(self, item: int):
        frames = []

        for i in range(self.config.number_of_prev_frames):
            frames.append(self.get_sim_state(i + item))
        x = torch.cat(frames, 0).flatten()
        y = self.get_sim_state(
            self.config.number_of_prev_frames + item).flatten()

        return x, y


def generate_training_data(
        queue: Queue,
        seed: int,
        lateral_acceleration,
        vertical_acceleration,
        number_of_balls,
        ball_lat_potential,
        ball_vert_potential,
        randomize_potential,
        number_of_prev_frames,
        model_interval,
        simulation_interval
):
    torch.manual_seed(seed)

    gravity = torch.tensor((
        lateral_acceleration,
        vertical_acceleration
    ),
        dtype=torch.float32)
    sim = Simulation(None, None, gravity, device='cpu')
    sim.add_random_ball_potential(
        n=number_of_balls,
        lat_potential=ball_lat_potential,
        vert_potential=ball_vert_potential,
        randomize_potential=randomize_potential
    )

    initial = []
    for _ in range(number_of_prev_frames):
        for _ in range(model_interval):
            sim.advance(simulation_interval)
        initial.append(deepcopy(sim.board))
    frame_cache = FrameCache(initial)
    while True:
        while not queue.full():
            for _ in range(model_interval):
                sim.advance(simulation_interval)
            x = frame_cache.get_nn_input()
            y = sim.board
            frame_cache.update(sim.board)
            queue.put((x, y))


class GeneratedBallDataset(torch.utils.data.Dataset):
    """
    Runs an internal simulation, creating training data as it is needed.

    """
    processes: List[Process]
    q: Queue

    length: int
    config: Config
    device: str

    def __init__(self, config: Config, length: int, device: str, num_procs: int):
        self.q = Queue(400)
        self.processes = []

        for i in range(num_procs):
            # Have to break out values coz u can't pass cuda objects to
            # sub-processes.
            p = Process(
                    target=generate_training_data,
                    args=(
                        self.q,
                        config.seed + i,
                        config.lateral_acceleration,
                        config.vertical_acceleration,
                        config.number_of_balls,
                        config.ball_lat_potential,
                        config.ball_vert_potential,
                        config.randomize_potential,
                        config.number_of_prev_frames,
                        config.model_interval,
                        config.simulation_interval
                    )
                )
            p.start()
            self.processes.append(p)

        self.device = device
        self.length = length
        self.config = config

    def __len__(self):
        return self.length

    def __getitem__(self, item: int):
        x, y = self.q.get()
        return (x.to(self.config.device).to(torch.float32).flatten(),
                y.to(self.config.device).to(torch.float32).flatten())



def train(config: Config):
    torch.manual_seed(config.seed)

    # Matrix needed if training process is to be visualized.
    if config.visualize_training:
        matrix = LEDMatrix(
            config.matrix_width,
            config.matrix_length,
            config.pallete_size
        )

    # Configure dataset for on-the-fly generation or precomputed.
    if config.gen_during_training:
        train_data = GeneratedBallDataset(
            config,
            config.number_of_examples,
            config.device,
            1
        )
    else:
        train_data = StoredBallDataset(
            config,
            config.device,
            config.get_data_path(),
            config.number_of_examples
        )

    # Attempt to load cached model.
    last_save = datetime.datetime.now()
    if config.model_caching and config.get_cached_model_path().exists():
        config.model = torch.load(config.get_cached_model_path())

    config.model.train()
    config.model.to(config.device)

    # Training loop.
    for e in range(config.epochs):

        # Reduce batch size to 1 if visualizing this epoch.
        if not config.visualize_training or e in config.skip_epochs:
            train_loader = torch.utils.data.DataLoader(train_data,
                                                       config.batch_size,
                                                       shuffle=False)
        else:
            train_loader = torch.utils.data.DataLoader(train_data,
                                                       1,
                                                       shuffle=False)

        t_hits = 0
        t_total = 0
        loss_total = 0
        print(f'Starting Epoch: {e}')
        pbar_iter = tqdm(train_loader)
        for X, Y in pbar_iter:
            pred = config.model(X)

            # Adjust target probs for multiple balls.
            if config.number_of_balls > 1:
                Y_probs = Y / config.number_of_balls
            else:
                Y_probs = Y

            # The usual.
            loss = config.loss_fxn(pred, Y_probs)
            loss.backward()
            config.optimizer.step()
            config.optimizer.zero_grad()

            predicted_frames = pred_to_frame(pred, False, config.number_of_balls)
            hits = (Y.to(torch.bool) & predicted_frames).sum()

            # Visualize prediction and target.
            if config.visualize_training and e not in config.skip_epochs:
                matrix.set_layer('target',
                                 Y[0, :])
                probs_heatmap = pred_to_heatmap(
                    pred[0, :], config.heatmap_palette_size,
                    config.heatmap_palette_offset, True
                )
                matrix.set_layer(
                    'prediction',
                    probs_heatmap
                )
                time.sleep(1 / config.simulation_fps)

            t_hits += hits
            t_total += config.batch_size * config.number_of_balls
            loss_total += loss.sum().item()
            pbar_iter.set_description(
                f'acc: {hits:02d}/{config.batch_size * config.number_of_balls:02d} '
                f'loss: {loss.sum().item():4.2f}')

        print(f'Epoch Train Accuracy: {t_hits / t_total * 100: .2f}%\n'
              f'Mean Loss :{loss_total / config.number_of_examples: .2f}')

        if config.scheduler:
            config.scheduler.step()

        # Save model intermediary if required.
        seconds_since_last_cache = (
                    datetime.datetime.now() - last_save).total_seconds()
        if config.model_caching and seconds_since_last_cache > 60:
            torch.save(config.model, config.get_cached_model_path())

    torch.save(config.model, config.get_model_path())
