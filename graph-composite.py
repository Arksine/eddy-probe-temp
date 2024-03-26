#!/usr/bin/env python3
# Eddy Probe Drift Data Collection
#
# Copyright (C) 2024 Eric Callahan <arksine.code@gmail.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
from __future__ import annotations
import argparse
import pathlib
import re
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial

def poly_to_str(coef):
    out = "y(x) = "
    deg = len(coef) - 1
    for i, c in enumerate(reversed(coef)):
        if int(c) == c:
            c == int(c)
        if c == 0:
            continue
        cur_deg = deg - i
        x_str = f"x^{cur_deg}" if cur_deg else ""
        if i == 0:
            out = f"{out}{c}{x_str}"
        else:
            sym = "-" if c < 0 else "+"
            out = f"{out} {sym} {abs(c)}{x_str}"
    return out

def plot_saved_data(dpath: pathlib.Path):
    desc_m = re.match(r"data-samples-(.+)\.json", dpath.name)
    if desc_m is None:
        desc = dpath.name
    else:
        desc = desc_m.group(1)
    data = json.loads(dpath.read_text())
    smp = [s for s in data["samples"] if s[0] < 85.]
    x, y = np.array(smp).transpose()
    poly: Polynomial = Polynomial.fit(x, y, 1)
    px, py = poly.linspace()
    coef = poly.convert().coef
    plt.plot(x, y, ".", label=desc)
    plt.plot(px, py, "-", label=f"y = {round(coef[1], 4)}x")
    print(f"{desc} polynomial: {poly_to_str(coef)}")

def data_file_sorter(path: pathlib.Path):
    match = re.match(r"data-samples-(\d+)c-((?:\d+mm)|(?:base)).+json", path.name)
    if match is None:
        return tuple([ord(c) for c in path.name])
    btmp = int(match.group(1))
    height = match.group(2)
    if height == "base":
        h = 0
    else:
        h = int(height[:-2])
    return (h, btmp)

def main():
    parser = argparse.ArgumentParser(description="Plot Composite Graph")
    parser.add_argument(
        "-i", "--input-directory", type=pathlib.Path,
        default=pathlib.Path(__file__).parent,
        help="Path to input directory"
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output file path"
    )
    args = parser.parse_args()
    if args.output is not None:
        matplotlib.use("Agg")
    parent_dir = args.input_directory.expanduser().resolve()
    print(f"Loading files from folder {parent_dir}")
    pattern = "data-samples*.json"
    paths = sorted(parent_dir.glob(pattern), key=data_file_sorter)
    if not paths:
        print("No data sample files found")
        exit(1)
    for fpath in paths:
        plot_saved_data(fpath)
    plt.xlabel("Temperature")
    plt.ylabel("Frequency")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    fig.tight_layout()
    if args.output is None:
        plt.show()
    else:
        fig.savefig(args.output)


if __name__ == "__main__":
    main()
