#!/usr/bin/env python3

import driver
import sys
import subprocess
import numpy as np

# benchmarks = [
#     ('luminance.t', 'Luminance'),
#     ('codes/huber.n', 'Huber'),
#     ('codes/axbench/blackscholes-2.n', 'BlackScholes'),
#     ('codes/kelvinToXY.n', 'Camera'),
#     ('codes/quake_phi_phi.n', 'EQuake'),
#     ('codes/axbench/jmeint.n', 'Jmeint'),
# ]


benchmarks = [
    ('synthetic/complexity_matters.n', 'Complexity Matters'),
    ('synthetic/frequency_matters.n', 'Frequency Matters'),
    ('synthetic/analysis_overcomplicates.n', 'Analysis Overcomplicates'),
    ('synthetic/analysis_nonuniformity.n', 'Analysis Nonuniformity'),
    ('synthetic/analysis_bad.n', 'Analysis Bad'),
]

def main():


    print(r'\multirow{2}{1.5cm}{\bf Benchmark} & \multirow{2}{1.5cm}{\centering\textbf{Path}} & \multirow{2}{2cm}{\centering\textbf{Complexity}} & \multirow{2}{2cm}{\centering\bf Frequency \\ Distribution} & \multirow{2}{2cm}{\centering\bf Uniform \\ Distribution} & \multirow{2}{2cm}{\centering\bf Complexity \\ Distribution} \\&&&&\\ \midrule')

    for i, (code, name) in enumerate(benchmarks):
        output = subprocess.check_output([sys.executable, 'plot.py', '--program', code, '--no-theo', '--print-table'], stderr=subprocess.DEVNULL, universal_newlines=True)
        print(r'\multirow{{{}}}{{1.5cm}}{{\bf\centering {}}}'.format(len(output.splitlines()), name))
        print('\n'.join('& {}'.format(line) for line in output.splitlines()))
        if i != len(benchmarks) - 1:
            print(r'\midrule')

    print(r'\bottomrule')
    print('\n\n\n')

    # return

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
        # freq_empirical = 1/(1 + freq_empirical)  - 1

        freq_theoretical = parse_line(output, 'improvement over frequency (theoretical):')
        # freq_theoretical = 1/(1 + freq_theoretical)  - 1

        uniform_empirical = parse_line(output, 'improvement over uniform:')
        # uniform_empirical = 1/(1 + uniform_empirical)  - 1

        uniform_theoretical = parse_line(output, 'improvement over uniform (theoretical):')
        # uniform_theoretical = 1/(1 + uniform_theoretical)  - 1

        p = driver.read_program(code)
        distribution = p.config.distribution

        data[name] = (freq_empirical, freq_theoretical, uniform_empirical, uniform_theoretical, distribution)

        print('{} & ${}$ & ${}$ & ${:.2%}$ & ${:.2%}$ & ${:.2%}$ & ${:.2%}$ \\\\'.format(name, lines_of_code, len(distribution), freq_theoretical, freq_empirical, uniform_theoretical, uniform_empirical).replace('%', r'\%'))

    print(r'\midrule')

    # Geo Mean
    print('Geomean & & & ${:.2%}$ & ${:.2%}$ & ${:.2%}$ & ${:.2%}$ \\\\'.format(
        np.exp(np.mean([np.log(1 + data[name][1]) for name in data])) - 1,
        np.exp(np.mean([np.log(1 + data[name][0]) for name in data])) - 1,
        np.exp(np.mean([np.log(1 + data[name][3]) for name in data])) - 1,
        np.exp(np.mean([np.log(1 + data[name][2]) for name in data])) - 1,
    ).replace('%', r'\%'))

    print(r'\bottomrule')

if __name__ == '__main__':
    main()
