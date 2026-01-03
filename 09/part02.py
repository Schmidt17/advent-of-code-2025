from pathlib import Path
import fire
import numpy as np
from itertools import combinations
from scipy import sparse


def get_slice(coords0, coords1):
    x0, y0 = coords0
    x1, y1 = coords1

    is_hori = (y0 == y1)

    if is_hori:
        start = min(x0, x1)
        end = max(x0, x1)
        const = y0
        if x1 > x0:
            direction = 0
        else:
            direction = 2
    else:
        start = min(y0, y1)
        end = max(y0, y1)
        const = x0
        if y1 > y0:
            direction = 1
        else:
            direction = 3

    return direction, start, end, const


def mark_border_line(coords0, coords1, tile_map, value=1):
    direction, start, end, const = get_slice(coords0, coords1)

    if direction in (0, 2):
        tile_map[const, start: end] = value
    else:
        tile_map[start: end, const] = value

    return direction


def get_turn(direction0, direction1):
    diff = (direction1 - direction0) % 4

    if diff == 0:
        return 0

    if diff == 1:
        return 0.25

    if diff == 3:
        return -0.25

    if diff == 2:
        raise ValueError(f"Found 180° turn, turning number ambiguous")


def mark_border(coords: np.ndarray, tile_map) -> float:
    turning_number = 0.0
    last_direction = None
    directions = []

    for idx in range(coords.shape[0] - 1):
        direction = mark_border_line(coords[idx], coords[idx + 1], tile_map)

        if len(directions) > 0:
            turning_number += get_turn(directions[-1], direction)

        directions.append(direction)

    direction = mark_border_line(coords[-1], coords[0], tile_map)
    turning_number += get_turn(directions[-1], direction)
    directions.append(direction)

    turning_number += get_turn(direction, directions[0])

    return turning_number, directions


def mark_outside_border(coords: np.ndarray, tile_map, orientation, directions):
    for idx in range(coords.shape[0] - 1):
        c0 = coords[idx].copy()
        c1 = coords[idx + 1].copy()

        if directions[idx] == 0:
            if orientation == "cw":
                c0[0] += -1
                c1[0] += -1
            else:
                c0[0] += 1
                c1[0] += 1

        elif directions[idx] == 1:
            if orientation == "cw":
                c0[1] += -1
                c1[1] += -1
            else:
                c0[1] += 1
                c1[1] += 1

        elif directions[idx] == 2:
            if orientation == "cw":
                c0[0] += 1
                c1[0] += 1
            else:
                c0[0] += -1
                c1[0] += -1

        elif directions[idx] == 3:
            if orientation == "cw":
                c0[1] += 1
                c1[1] += 1
            else:
                c0[1] += -1
                c1[1] += -1

        _ = mark_border_line(c0, c1, tile_map, value=2)




def main(input_path: str):
    input_path = Path(input_path)
    coords = np.loadtxt(input_path, delimiter=",", dtype=int)

    assert np.min(coords) > 0, "Shape has no margin towards lower indices"

    max_x, max_y = np.max(coords, axis=0)
    tile_map = sparse.lil_array((max_y + 2, max_x + 2), dtype=np.uint8)

    turning_number, directions = mark_border(coords, tile_map)

    if turning_number == 1.0:
        orientation = "ccw"
    elif turning_number == -1.0:
        orientation = "cw"
    else:
        raise ValueError(f"Turning number must be 1.0 or -1.0, got {turning_number}")

    # print(tile_map.count_nonzero(), orientation)

    mark_outside_border(coords, tile_map, orientation, directions)

    print(tile_map.toarray())

    # n_red = coords.shape[0]
    # max_area = 0
    # for i, j in combinations(range(n_red), r=2):
    #     area = np.prod(np.abs(coords[i] - coords[j] + 1))
    #     if area > max_area:
    #         max_area = area

    # print(f"{max_area=}")


if __name__ == '__main__':
    fire.Fire(main)
