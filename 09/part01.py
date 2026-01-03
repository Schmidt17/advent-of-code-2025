from pathlib import Path
import fire
import numpy as np
from itertools import combinations


def main(input_path: str):
    input_path = Path(input_path)
    coords = np.loadtxt(input_path, delimiter=",", dtype=int)

    n_red = coords.shape[0]

    max_area = 0
    for i, j in combinations(range(n_red), r=2):
        area = np.prod(np.abs(coords[i] - coords[j] + 1))
        if area > max_area:
            max_area = area

    print(f"{max_area=}")


if __name__ == '__main__':
    fire.Fire(main)
