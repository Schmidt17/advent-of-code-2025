import re
import fire
from pathlib import Path


dir_to_sign = {
    "L": -1,
    "R": 1
}


def parse_rotation(rot_str: str) -> int:
    m = re.match(r"(?P<dir>[LR])(?P<num>\d+)", rot_str)

    if m is None:
        raise ValueError(f'Could not parse rotation from string "{rot_str}"')

    direction = m.group("dir")
    n_clicks = int(m.group("num"))

    return dir_to_sign[direction] * n_clicks


def load_lines(filepath: Path) -> list[str]:
    with open(filepath, 'r') as f:
        content = f.read()

    return content.splitlines()


def main(input_path: str):
    input_path = Path(input_path)
    lines = load_lines(input_path)

    rots = map(parse_rotation, lines)

    zero_counter = 0
    current = 50

    for rot in rots:
        current = (current + rot) % 100

        if current == 0:
            zero_counter += 1

    print(f"The password is: {zero_counter}")


if __name__ == '__main__':
    fire.Fire(main)
