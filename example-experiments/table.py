#!/usr/bin/env python3

import sys
import subprocess
import numpy as np

benchmarks = [
    ('codes/quake_phi_phi.n', 'Quake'),
    ('codes/camera.n', 'Camera'),
    ('codes/huber.n', 'Huber'),
    ('codes/axbench/blackscholes-2.n', 'BlackScholes'),
    ('codes/axbench/jmeint.n', 'Jmeint'),
]

def main():
    data = {}
    def parse_line(text, string):
        lines = text.splitlines()
        for line in lines:
            if string in line:
                return float(line.split()[-1])

    print('Benchmark & Frequency & Frequency (theoretical) & Uniform & Uniform (theoretical) \\\\')

    for (code, name) in benchmarks:
        output = subprocess.check_output([sys.executable, 'plot.py', '--program', code, '--no-theo', '--no-show'], stderr=subprocess.DEVNULL, universal_newlines=True)
        freq_empirical = parse_line(output, 'improvement over frequency:')
        freq_theoretical = parse_line(output, 'improvement over frequency (theoretical):')
        uniform_empirical = parse_line(output, 'improvement over uniform:')
        uniform_theoretical = parse_line(output, 'improvement over uniform (theoretical):')

        data[name] = (freq_empirical, freq_theoretical, uniform_empirical, uniform_theoretical)

        print('{} & {:.2%} & {:.2%} & {:.2%} & {:.2%} \\\\'.format(name, freq_empirical, freq_theoretical, uniform_empirical, uniform_theoretical))

    print(r'\hline')
    # Euclidean Mean
    print('Average (euclidean) & {:.2%} & {:.2%} & {:.2%} & {:.2%} \\\\'.format(
        sum([data[name][0] for name in data]) / len(data),
        sum([data[name][1] for name in data]) / len(data),
        sum([data[name][2] for name in data]) / len(data),
        sum([data[name][3] for name in data]) / len(data),
    ))

    # Geo Mean
    print('Average (geometric) & {:.2%} & {:.2%} & {:.2%} & {:.2%} \\\\'.format(
        np.exp(np.mean([np.log(1 + data[name][0]) for name in data])) - 1,
        np.exp(np.mean([np.log(1 + data[name][1]) for name in data])) - 1,
        np.exp(np.mean([np.log(1 + data[name][2]) for name in data])) - 1,
        np.exp(np.mean([np.log(1 + data[name][3]) for name in data])) - 1,

    ))

if __name__ == '__main__':
    main()
