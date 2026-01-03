from pathlib import Path
import fire
import numpy as np
from itertools import combinations
from scipy import sparse
from tqdm import tqdm
import time


neig_dists = np.array([
    [0, 1],
    [-1, 1],
    [-1, 0],
    [-1, -1],
    [0, -1],
    [1, -1],
    [1, 0],
    [1, 1]
])


def mark_outer_border(tile_map: sparse.lil_array, start_tile: np.ndarray):
    bin_map = tile_map.tocsr()

    # convert tile coord into (row, col) format by reading it backwards:
    start_indices = start_tile[::-1]

    rows, cols = tile_map.shape

    neighs = get_neighbors(start_indices, rows, cols)
    open_tiles = [(start_indices, neighs)]
    it_counter = 0

    while len(open_tiles) > 0:
        if it_counter % 1000 == 0:
            print(f"{it_counter}                   ", end="\r")
        current, neighs = open_tiles.pop()
        tile_map[current[0], current[1]] = 2

        for n in neighs:
            if tile_map[n[0], n[1]] == 0:
                row_st, row_end = max(n[0] - 1, 0), min(n[0] + 2, rows)
                col_st, col_end = max(n[1] - 1, 0), min(n[1] + 2, cols)
                touches_border = bin_map[row_st: row_end, col_st: col_end].count_nonzero() > 0

                # only add n if it touches the border
                if touches_border:
                    nns = get_neighbors(n, rows, cols)
                    open_tiles.append((n, nns))

        it_counter += 1


def get_neighbors(idx, rows, cols):
    row, col = idx

    neighs = idx + neig_dists

    accepted = np.all(neighs >= 0, axis=1) * (neighs[:, 0] < rows) * (neighs[:, 1] < cols)

    return neighs[accepted]


def mark_border(tile_map, coords):
    for idx in range(coords.shape[0] - 1):
        mark_border_line(coords[idx], coords[idx + 1], tile_map)

    # close the shape by connecting to the first tile
    mark_border_line(coords[-1], coords[0], tile_map)


def get_slice(coords0, coords1):
    x0, y0 = coords0
    x1, y1 = coords1

    is_hori = (y0 == y1)

    if is_hori:
        start = min(x0, x1)
        # inclusive end, to avoid inconsistency of
        # left- and right-going lines
        end = max(x0, x1) + 1
        const = y0

    else:
        start = min(y0, y1)
        # inclusive, sim. to above
        end = max(y0, y1) + 1
        const = x0

    return is_hori, start, end, const


def get_rect_slice(coords0, coords1):
    x0, y0 = coords0
    x1, y1 = coords1

    xmin = min(x0, x1)
    xmax = max(x0, x1)
    ymin = min(y0, y1)
    ymax = max(y0, y1)

    return ymin, ymax + 1, xmin, xmax + 1


def mark_border_line(coords0, coords1, tile_map, value=1):
    is_hori, start, end, const = get_slice(coords0, coords1)

    if is_hori:
        tile_map[const, start: end] = value
    else:
        tile_map[start: end, const] = value


def main(input_path: str):
    input_path = Path(input_path)
    coords = np.loadtxt(input_path, delimiter=",", dtype=int)

    # Make sure that we have some space to draw a border around the shape.
    # We only need to check small indices, since we can just add a spacer
    # towards higher indices, without needing to transform coordinates
    assert np.min(coords) > 0, "Shape has no margin towards lower indices"

    max_x, max_y = np.max(coords, axis=0)
    tile_map = sparse.lil_array((max_y + 2, max_x + 2), dtype=np.uint8)

    # mark the border defined by the input coordinates (inplace)
    mark_border(tile_map, coords)

    print("Border tiles:", tile_map.count_nonzero())

    # Mark a 1-tile wide outline around the border, to define the
    # onset of the outside.
    # Start with a tile that is definitely on the outside and touches the border,
    # e.g. one beyond one of those with maximal y-coordinate (choose the first one here):
    max_y_idx = np.atleast_1d(np.argmax(coords[:, 1]))[0]
    start_tile = coords[max_y_idx].copy()
    start_tile[1] += 1

    t0 = time.perf_counter()
    mark_outer_border(tile_map, start_tile)  # (inplace)
    t1 = time.perf_counter()
    print(f"Marking took {t1 - t0} s")

    print("Border + outer border tiles:", tile_map.count_nonzero())

    # convert to csr, since we don't need to modify the tile map anymore
    # also, only keep the outer border, we don't need the original anymore
    # speeds up the checks below
    tile_map = (tile_map == 2).tocsr()

    n_red = coords.shape[0]
    max_area = 0
    max_coords = None

    # compute areas of all possible rectangles
    index_combinations = np.array(list(combinations(range(n_red), r=2)))
    areas = np.empty(index_combinations.shape[0], dtype=float)
    for comb_idx in range(index_combinations.shape[0]):
        i, j = index_combinations[comb_idx]
        areas[comb_idx] = np.prod(np.abs(coords[i] - coords[j] + 1))

    # sort the rectangles by area
    sort_inds = np.argsort(areas)

    # check rectangles for acceptance in order of descending area
    for comb_idx in tqdm(sort_inds[::-1]):
        i, j = index_combinations[comb_idx]

        # check whether the rectangle touches the outer border
        row_st, row_end, col_st, col_end = get_rect_slice(coords[i], coords[j])
        map_slice = tile_map[row_st: row_end, col_st: col_end]
        touches_border = map_slice.count_nonzero() > 0

        # the first one that is valid will be the biggest one:
        if not touches_border:
            max_area = areas[comb_idx]
            max_coords = (coords[i], coords[j])

            break

    # print(areas)
    # exit()
    # for i, j in tqdm(combinations(range(n_red), r=2)):
    #     # check whether the rectangle touches the outer border
    #     row_st, row_end, col_st, col_end = get_rect_slice(coords[i], coords[j])
    #     map_slice = tile_map[row_st: row_end, col_st: col_end]
    #     touches_border = (map_slice == 2).count_nonzero() > 0

    #     if not touches_border:
    #         area = np.prod(np.abs(coords[i] - coords[j] + 1))
    #         if area > max_area:
    #             max_area = area
    #             max_coords = (coords[i], coords[j])

    print(f"{max_area=}")
    print(f"{max_coords=}")


if __name__ == '__main__':
    fire.Fire(main)
