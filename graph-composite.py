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
from typing import List

def poly_to_str(coefs):
    parts = ["y(x) = "]
    deg = len(coefs) - 1
    for i, coef in enumerate(reversed(coefs)):
        if round(coef, 8) == int(coef):
            coef == int(coef)
        if abs(coef) < 1e-10:
            continue
        cur_deg = deg - i
        x_str = "x^%d" % (cur_deg,) if cur_deg > 1 else "x" * cur_deg
        if len(parts) == 1:
            parts.append("%f%s" % (coef, x_str))
        else:
            sym = "-" if coef < 0 else "+"
            parts.append("%s %f%s" % (sym, abs(coef), x_str))
    return " ".join(parts)

def plot_saved_data(dpath: pathlib.Path):
    desc_m = re.match(r"data-samples-(.+)\.json", dpath.name)
    if desc_m is None:
        desc = dpath.name
    else:
        desc = desc_m.group(1)
    data = json.loads(dpath.read_text())
    smp = [s[:2] for s in data["samples"] if 30. < s[0] < 80.]
    x, y = np.array(smp).transpose()
    poly: Polynomial = Polynomial.fit(x, y, 2)
    px, py = poly.linspace()
    coef = poly.convert().coef
    plt.plot(x, y, ".", label=desc)
    plt.plot(px, py, "-", label=f"y = {round(coef[1], 4)}x")
    print(f"{desc} polynomial: {poly_to_str(coef)}")
    print(f"Coefs: {tuple(coef)}")

def plot_slope_data(pathlist: List[pathlib.Path]):
    start_temp = 0
    sample_list = []
    for fpath in pathlist:
        desc_m = re.match(r"data-samples-(.+)\.json", fpath.name)
        if desc_m is None:
            desc = fpath.name
        else:
            desc = desc_m.group(1)
        data = json.loads(fpath.read_text())
        smp = data["samples"]
        start_temp = max(start_temp, smp[0][0])
        sample_list.append((smp, desc))
    freqs = []
    coefs = []
    for samples, desc in sample_list:
        trimmed = [s[:2] for s in samples if start_temp < s[0] < 85.]
        if len(trimmed) < 3:
            continue
        base_freq = trimmed[0][1]
        deltas = [(t - start_temp, base_freq - f) for t, f in trimmed]
        x, y = np.array(deltas).transpose()
        poly: Polynomial = Polynomial.fit(x, y, 1)
        coef = poly.convert().coef
        freqs.append(base_freq)
        coefs.append(coef[1])

    spoly: Polynomial = Polynomial.fit(freqs, coefs, 1)
    sx, sy = spoly.linspace()
    print(f"Min Temp: {start_temp}")
    print(f"Slope Range: {sy.min()}, {sy.max()}")
    print(f"Freq Range: {sx.min()}, {sx.max()}")
    print(f"Slope polynomial: {poly_to_str(spoly.convert().coef)}")
    plt.plot(freqs, coefs, ".")
    plt.plot(sx, sy, "-")
    plt.xlabel("Frequency Delta from Min")
    plt.ylabel("Slope")

def plot_error(pathlist: List[pathlib.Path]):
    start_temp = 0
    sample_list = []
    for fpath in pathlist:
        desc_m = re.match(r"data-samples-(.+)\.json", fpath.name)
        if desc_m is None:
            desc = fpath.name
        else:
            desc = desc_m.group(1)
        data = json.loads(fpath.read_text())
        smp = data["samples"]
        start_temp = max(start_temp, smp[0][0])
        sample_list.append((smp, desc))

    for (samples, desc) in sample_list:
        trimmed = [s[:2] for s in samples if start_temp < s[0] < 80.]
        if len(trimmed) < 3:
            continue
        base_freq = trimmed[0][1]
        deltas = [(t, base_freq - f) for t, f in trimmed]
        x, y = np.array(deltas).transpose()
        poly: Polynomial = Polynomial.fit(x, y, 2)
        coef = poly.convert().coef
        px, py = poly.linspace()
        plt.plot(x, y, ".", label=desc)
        plt.plot(px, py, "-", label=f"y = {round(coef[1], 4)}x")
        print(f"{desc} polynomial: {poly_to_str(coef)}")
    plt.xlabel("Temperature")
    plt.ylabel("Frequency Error")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

def plot_drift_dump(path: pathlib.Path, args: argparse.Namespace):
    pd_data = json.loads(path.read_text())
    freqs = []
    coefs = []
    for i, samples in enumerate(pd_data["data"]):
        height = .05 + i * .5
        base_temp, base_freq = samples[0]
        if args.plot_error:
            deltas = [(t - base_temp, base_freq - f) for t, f in samples]
            x, y = np.array(deltas).transpose()
        else:
            x, y = np.array(samples).transpose()
        poly = Polynomial.fit(x, y, 2)
        coef = poly.convert().coef
        px, py = poly.linspace()
        if not args.plot_slope:
            plt.plot(x, y, ".", label=f"Height: {height}")
            plt.plot(px, py, "-", label=f"y = {round(coef[1], 4)}x")
        freqs.append(base_freq)
        coefs.append(coef[1])
        print(f"Error Poly at height {height}: {poly_to_str(coef)}")
    if args.plot_slope:
        spoly = Polynomial.fit(freqs, coefs, 1)
        sx, sy = spoly.linspace()
        plt.plot(freqs, coefs, ".")
        plt.plot(sx, sy, "-")
    else:
        plt.xlabel("Temperature Delta" if args.plot_error else "Temperature")
        plt.ylabel("Frequency Error" if args.plot_error else "Frequency")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

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
    parser.add_argument(
        "-s", "--plot-slope", action="store_true",
        help="Plot slope"
    )
    parser.add_argument(
        "-e", "--plot-error", action="store_true",
        help="Plot slope"
    )
    parser.add_argument(
        "-d", "--drift-dump", default=None,
        help="Plot drift dump file"
    )
    args = parser.parse_args()
    if args.output is not None:
        matplotlib.use("Agg")
    if args.drift_dump is not None:
        dd_path = pathlib.Path(args.drift_dump).expanduser().resolve()
        plot_drift_dump(dd_path, args)
    else:
        parent_dir = args.input_directory.expanduser().resolve()
        print(f"Loading files from folder {parent_dir}")
        pattern = "data-samples*.json"
        paths = sorted(parent_dir.glob(pattern), key=data_file_sorter)
        if not paths:
            print("No data sample files found")
            exit(1)
        if args.plot_slope:
            plot_slope_data(paths)
        elif args.plot_error:
            plot_error(paths)
        else:
            for fpath in paths:
                plot_saved_data(fpath)
            plt.xlabel("Temperature")
            plt.ylabel("Frequency Error")
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
