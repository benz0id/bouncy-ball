import dataclasses
import sys
from asyncio import sleep
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from threading import Thread
from typing import List, Tuple, Dict, Union

import serial
import torch
import time
from serial.serialutil import SerialException
from multiprocessing import Queue
import multiprocessing as mp
import traceback

DISPLAY_HEIGHT = 32
DISPLAY_WIDTH = 64


class Process(mp.Process):
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            mp.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            raise e  # You can still rise this exception if you need to

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


def _receive_layers(q: Queue, num_colours: int):
    for port in ['/dev/tty.usbmodem101', '/dev/tty.usbmodem1101']:
        s = None
        try:
            s = serial.Serial(port)
        except SerialException:
            continue
    if not port:
        raise SerialException('Could not find LED matrix port.')
    while True:
        layers = q.get()
        _update_from_layers(layers, s, num_colours)


def _update_from_layers(layers: Dict[str, torch.tensor],
                        s: serial.Serial, max_palette: int):
    msg = []
    for j in range(DISPLAY_HEIGHT - 1, -1, -1):
        for i in range(DISPLAY_WIDTH):
            byte = 0
            for layer in layers:
                byte = layers[layer][i, j].item()
                if byte:
                    if not 0 <= byte < max_palette + 2:
                        raise ValueError(f'{byte} is not a valid colour for'
                                         f' a pallete of size'
                                         f' {max_palette}. Received in'
                                         f'layer {layer}.')
                    if byte > 2:
                        byte += 2
                    break

            msg.append(int(byte).to_bytes(1))
    s.write(b''.join(msg))
    s.readline().decode()


class LEDMatrix:

    def __init__(self,
                 width: int,
                 height: int,
                 palette_size: int,
                 fps: int = 50,
                 run_update_loop: bool = True):
        self.width = width
        self.height = height
        self.num_colours = palette_size
        self.layers = {}
        self.fps = fps
        self.run_update_loop = run_update_loop
        if run_update_loop:
            self.q = Queue(20)
            self.writer_proc = Process(target=_receive_layers,
                                       args=[self.q, self.num_colours])
            self.writer_proc.start()

            self.matrix_thread = Thread(target=self.matrix_update_loop)
            self.matrix_thread.start()
        else:
            self.config_port()

    def config_port(self):

        for port in ['/dev/tty.usbmodem101', '/dev/tty.usbmodem1101']:
            self.s = None
            try:
                self.s = serial.Serial(port)
            except SerialException:
                continue
        if not port:
            raise SerialException('Could not find LED matrix port.')

    def set_layer(self, name: str, array: torch.tensor) -> None:
        if tuple(array.shape) == tuple([self.width * self.height]):
            array = array.view(self.width, self.height)
        elif tuple(array.shape) == tuple([self.width, self.height]):
            pass
        else:
            raise ValueError(f'Input array \'{name}\' has invalid shape.'
                             f' {array.shape} != '
                             f'{tuple([self.width, self.height])}')
        self.layers[name] = array

    bad_bytes = [3, 4]

    def update_panel(self):
        if self.run_update_loop:
            if not self.q.full():
                for layer in self.layers:
                    self.layers[layer] = self.layers[layer].cpu()
                self.q.put(self.layers)
        else:
            _update_from_layers(self.layers, self.s, self.num_colours)

    def matrix_update_loop(self):
        while True:
            if not self.q.full():
                cpu_layers = {}
                for layer in self.layers:
                    cpu_layers[layer] = self.layers[layer].cpu()
                self.q.put(cpu_layers)
            time.sleep(1 / self.fps)
            if self.writer_proc.exception:
                error, traceback = self.writer_proc.exception
                print(traceback)


def to_range(val: float, min_val: int, max_val: int) -> int:
    if val < min_val:
        return min_val
    if val > max_val:
        return max_val
    else:
        return int(val)


@dataclasses.dataclass
class Ball:
    id: int
    position: torch.tensor
    velocity: torch.tensor

    def boundary_check(self):
        inv_x = False
        inv_y = False

        x_high = self.position[0].item() > DISPLAY_WIDTH
        x_low = 0 > self.position[0].item()
        going_right = self.velocity[0] > 0

        y_high = self.position[1].item() > DISPLAY_HEIGHT
        y_low = 0 > self.position[1].item()
        going_up = self.velocity[1] > 0

        if x_high and going_right:
            inv_x = True
        if x_low and not going_right:
            inv_x = True

        if y_high and going_up:
            inv_y = True
        if y_low and not going_up:
            inv_y = True

        if inv_y:
            self.velocity *= torch.tensor([1, -1], dtype=torch.float32)
        if inv_x:
            self.velocity *= torch.tensor([-1, 1], dtype=torch.float32)


class Simulation:
    board: torch.tensor
    balls: List[Ball]

    gravity: torch.tensor

    iteration: int

    out_dir: Path

    id_iter: int

    def __init__(self, out_dir: Path, matrix: LEDMatrix,
                 gravity: torch.tensor,
                 device: str = 'cpu'):
        self.board = torch.zeros((DISPLAY_WIDTH, DISPLAY_HEIGHT),
                                 dtype=torch.bool, device=device)
        self.display = torch.zeros((DISPLAY_WIDTH, DISPLAY_HEIGHT),
                                   dtype=torch.float32, device=device)

        self.balls = []
        self.out_dir = out_dir

        self.gravity = gravity

        self.iteration = 0
        self.id_iter = 0

        if matrix:
            self.matrix = matrix
            self.matrix.set_layer('sim', self.board)

    def add_random_ball_velocity(self, n: int = 1,
                        starting_pos: Union[Tuple[int, int], None] = None,
                        lat_velocity: float = 5,
                        vert_velocity: float = 5,
                        randomize_velocity: bool = False):
        for _ in range(n):
            if starting_pos is None:
                pos = torch.rand(2) * torch.tensor([64, 32])
            else:
                pos = torch.tensor(starting_pos, dtype=torch.float32)

            v = torch.tensor([lat_velocity, vert_velocity],
                             dtype=torch.float32)
            if randomize_velocity:
                v *= torch.rand(2)

            self.balls.append(Ball(self.id_iter, pos, v))
            self.id_iter += 1

    def add_random_ball_potential(self, n: int = 1,
                        lat_potential: float = 5,
                        vert_potential: float = 5,
                        randomize_potential: bool = False):
        pot = torch.tensor([lat_potential, vert_potential],
                           dtype=torch.float32)
        if randomize_potential:
            pot *= torch.rand(2, dtype=torch.float32)

        for _ in range(n):
            dims = torch.tensor([64, 32])
            # Position relative to "ground"
            rel_g = self.gravity.abs()

            # Calculate position that would give maximum velocity potential less
            # that the total potential.
            max_pos = pot / rel_g
            max_pos[0] = min(max_pos[0].item(), 64)
            max_pos[1] = min(max_pos[1].item(), 32)
            rel_pos = torch.rand(2) * max_pos

            # Energy potential due to position.
            pos_potential = rel_pos * rel_g

            # Set velocity as remainder of potential.
            vel = (2 * (pot - pos_potential)).sqrt()

            # Convert back to absolute position and velocity.
            pos = (rel_pos - (dims * (self.gravity > 0))).abs()
            vel = vel * ((self.gravity < 0) * 2 - 1)

            pos[pos.isnan()] = 0
            vel[vel.isnan()] = 0

            self.balls.append(Ball(self.id_iter, pos, vel))
            self.id_iter += 1

    def save_board(self):
        torch.save(self.board, self.out_dir / f'{self.iteration}.pt')
        self.iteration += 1

    def _update_ball(self, ball: Ball, t: float) -> None:
        d = ball.velocity * t + 1 / 2 * self.gravity * t ** 2
        v = t * self.gravity
        ball.position += d
        ball.velocity += v

        ball.boundary_check()

    def _update_board(self, ball: Ball, add: bool):
        x, y = (to_range(ball.position[0], 0, 63),
                to_range(ball.position[1], 0, 31))
        if add:
            self.board[x, y] = True
            self.display[x, y] = ball.id
        else:
            self.board[x, y] = False
            self.display[x, y] = 0

    def advance(self, t: float) -> None:
        for ball in self.balls:
            self._update_board(ball, False)
            self._update_ball(ball, t)
            self._update_board(ball, True)
