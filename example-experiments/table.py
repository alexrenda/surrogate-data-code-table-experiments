#!/usr/bin/env python3

import driver
import sys
import subprocess
import numpy as np

benchmarks = [
    ('codes/huber.n', 'Huber'),
    ('codes/axbench/blackscholes-2.n', 'BlackScholes'),
    ('codes/camera.n', 'Camera'),
    ('codes/quake_phi_phi.n', 'Quake'),
    ('codes/axbench/jmeint.n', 'Jmeint'),
]

def main():
    data = {}
    def parse_line(text, string):
        lines = text.splitlines()
        for line in lines:
            if string in line:
                return float(line.split()[-1])

    print(r'\toprule')
    print(r'Benchmark & Lines of Code & Paths & Frequency (Predicted) & Frequency (Measured) & Uniform (Predicted) & Uniform (Measured) \\ \midrule')

    for (code, name) in benchmarks:
        with open(code, 'r') as f:
            lines_of_code = len(f.readlines())

        output = subprocess.check_output([sys.executable, 'plot.py', '--program', code, '--no-theo', '--no-show'], stderr=subprocess.DEVNULL, universal_newlines=True)

        freq_empirical = parse_line(output, 'improvement over frequency:')
        freq_empirical = 1/(1 + freq_empirical)  - 1

        freq_theoretical = parse_line(output, 'improvement over frequency (theoretical):')
        freq_theoretical = 1/(1 + freq_theoretical)  - 1

        uniform_empirical = parse_line(output, 'improvement over uniform:')
        uniform_empirical = 1/(1 + uniform_empirical)  - 1

        uniform_theoretical = parse_line(output, 'improvement over uniform (theoretical):')
        uniform_theoretical = 1/(1 + uniform_theoretical)  - 1

        p = driver.read_program(code)
        distribution = p.config.distribution

        data[name] = (freq_empirical, freq_theoretical, uniform_empirical, uniform_theoretical, distribution)

        print('{} & {} & {} & {:+.2%} & {:+.2%} & {:+.2%} & {:+.2%} \\\\'.format(name, lines_of_code, len(distribution), freq_theoretical, freq_empirical, uniform_theoretical, uniform_empirical))

    print(r'\midrule')

    # Geo Mean
    print('Geomean & & & {:.2%} & {:.2%} & {:.2%} & {:.2%} \\\\'.format(
        np.exp(np.mean([np.log(1 + data[name][1]) for name in data])) - 1,
        np.exp(np.mean([np.log(1 + data[name][0]) for name in data])) - 1,
        np.exp(np.mean([np.log(1 + data[name][3]) for name in data])) - 1,
        np.exp(np.mean([np.log(1 + data[name][2]) for name in data])) - 1,
    ))

    print(r'\bottomrule')

if __name__ == '__main__':
    main()
