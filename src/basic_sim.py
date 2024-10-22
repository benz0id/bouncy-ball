import time

from simulation import *
from pathlib import Path
import torch
from datetime import datetime

do_delay = True
UPS = 10
fps = None

interval = 1 / 10
save_every = 1
num_to_save = 10000
g = -9.81
out = Path('/single_ball_training')

# meter_to_unit = 32 / 0.19
# g = -9.834 * meter_to_unit

gravity = torch.tensor((-g, 0),
                       dtype=torch.float32)

def main():
    sim = Simulation(out, True,
                     gravity, fps)
    sim.add_random_ball(n=1)

    last = datetime.now()
    i = 0
    c = 0
    while i < num_to_save:
        # now = datetime.now()
        # sim.advance((now - last).total_seconds())
        # last = datetime.now()
        c += 1
        i += 1

        if c > save_every:
            # sim.save_board()
            c = 0
        sim.advance(interval)
        if do_delay:
            time.sleep(1 / UPS)

if __name__ == '__main__':
    main()