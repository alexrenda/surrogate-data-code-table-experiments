#!/usr/bin/env python

import argparse
import atexit
import math
import numpy as np
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import torch
import tqdm.auto as tqdm
import library
import driver

import plot_utils

import turaco

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt

C = np.array([
    (230, 159, 0),
    (86, 180, 233),
    (0, 158, 115),
    (240, 228, 66),
    (0, 114, 178),
]) / 255

_PYTHON = sys.executable
CURR_DIR = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
PAR_DIR = os.path.dirname(CURR_DIR)

delta = 0.1
delta_i = 1 - (1 - delta)**(1/3)

N_TRIALS = 10

mns = list(np.arange(1, 20)) + list(np.logspace(1, 5, 10).round().astype(int))
mns = np.logspace(1, 3, 10)
mns = sorted(np.array(list(mns)).round().astype(int))
mns = np.array(mns)

def get_theoretical_error(n_samples, complexity):
    n_samples = int(np.ceil(n_samples))
    C = 1
    return np.sqrt(C * (complexity + np.log(1/delta_i)) / n_samples)

def get_empirical_error(prog_name_base, n_samples, no_train=False):
    losses = []
    fnames = []
    procs = []

    if ':' in prog_name_base:
        prog_name_base, path_name = prog_name_base.split(':')
    else:
        path_name = 'all'

    new = False
    for i in range(N_TRIALS):
        fname = 'data/{prog_name_base}/{path_name}/{n}/{trial}/{lname}.loss'.format(prog_name_base=prog_name_base, n=n_samples,trial=i, path_name=path_name, lname=path_name if path_name != 'all' else 's')
        fnames.append(fname)
        if os.path.exists(fname):
            losses.append(torch.load(fname))
            continue
        elif no_train:
            continue

        new = True
        args = [_PYTHON, 'driver.py', '--program', prog_name_base,
             '--quiet',
             'train',
             '--n', str(n_samples), '--trial', str(i),
             '--lr', '5e-5', '--steps', '10000',
            ] + (['--path', path_name] if path_name != 'all' else [])

        proc = subprocess.Popen(
            args,
        )
        procs.append(proc)
    for p in procs:
        p.wait()
    for f in fnames:
        try:
            losses.append(torch.load(fname))
        except FileNotFoundError:
            if not no_train:
                raise

    losses.pop(np.argmax(losses))
    mid = np.mean(losses)
    lower = np.std(losses) / np.sqrt(len(losses))
    upper = np.std(losses) / np.sqrt(len(losses))

    return mid, lower, upper

def optimal_sampling(data_distribution, complexity):
    n = len(data_distribution)
    weights = {
        k: (data_distribution[k] * np.sqrt(complexity[k] + np.log(1/delta_i)))**(2/3)
        for k in data_distribution
    }

    probs = {
        k: v / sum(weights.values())
        for (k, v) in weights.items()
    }
    return probs

def uniform_sampling(data_distribution, complexity):
    return {k: 1/len(data_distribution) for k in data_distribution}

def data_dist_sampling(data_dist, complexities):
    return data_dist

def get_real_counts(sampling_method, data_dist, complexities):
    per_stratum_prob = sampling_method(data_dist, complexities)

    per_stratum_counts = [{
        k: int(np.ceil(v * n))
        for (k, v) in per_stratum_prob.items()
        } for n in mns
    ]

    return [sum(v.values()) for v in per_stratum_counts]

def get_theoretical_and_empirical_errors(sampling_method, data_dist, complexities, no_train=False):
    per_stratum_prob = sampling_method(data_dist, complexities)

    per_stratum_counts = [{
        k: int(np.ceil(v * n))
        for (k, v) in per_stratum_prob.items()
        } for n in mns
    ]

    per_stratum_theoretical_error = [{
        k: get_theoretical_error(v * n, complexities[k])
        for (k, v) in per_stratum_prob.items()
    } for n in mns]

    per_stratum_empirical_error = [{
        k: get_empirical_error(k, int(np.ceil(v * n)), no_train=no_train)
        for (k, v) in per_stratum_prob.items()
    } for n in tqdm.tqdm(mns)]

    theoretical_expectation_error = [
        sum(data_dist[k] * v for (k, v) in d.items())
        for d in per_stratum_theoretical_error
    ]

    empirical_expectation_error = [
        (
            sum(data_dist[k] * v[0] for (k, v) in d.items()),
            sum(data_dist[k] * v[1] for (k, v) in d.items()),
            sum(data_dist[k] * v[2] for (k, v) in d.items()),
        )
        for d in per_stratum_empirical_error
    ]

    return (
        per_stratum_prob,
        theoretical_expectation_error,
        empirical_expectation_error,
        per_stratum_counts,
        per_stratum_empirical_error,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--program', required=True, nargs='+')
    parser.add_argument('--print-dist', action='store_true')
    parser.add_argument('--no-train', action='store_true')
    parser.add_argument('--no-uniform', action='store_true')
    parser.add_argument('--no-theoretical', action='store_true')
    parser.add_argument('--no-optimal', action='store_true')
    parser.add_argument('--no-frequency', action='store_true')
    parser.add_argument('--no-analysis', action='store_true')
    parser.add_argument('--no-per-path', action='store_true')
    args = parser.parse_args()

    programs = [p.split(':')[0] for p in args.program]
    prog_data_dist = {
        p.split(':')[0]: (float(p.split(':')[1]) if len(p.split(':')) > 1 else 1)
        for p in args.program
    }

    data_dist = {}

    for p in programs:
        program = driver.read_program(p)
        for pth in program.config.distribution:
            data_dist['{}:{}'.format(p, pth)] = program.config.distribution[pth] * prog_data_dist[p]

        if os.path.exists('datasets/{p}'.format(p=p)):
            continue
        print('Generating dataset for {p}...'.format(p=p))
        datasets = library.collect_datasets(program, 100000)
        library.write_datasets(datasets, p)

    data_dist = {k: v / sum(data_dist.values()) for (k, v) in data_dist.items()}

    # complexity = {
    #     k: float(subprocess.run(
    #         [_PYTHON, '-m', 'turaco', '--program', k, 'complexity', '--input', str(0.06)],
    #         capture_output=True, text=True,
    #     ).stdout.strip().split('\n')[-1])
    #     for k in programs
    # }

    complexity = {}

    for p in programs:
        with open(p) as f:
            pg = turaco.parser.parse_program(f.read())

        all_paths = turaco.util.get_paths_of_program(pg)
        for (linp, path) in all_paths:
            col = turaco.main.calculate_complexity(linp, max_scaling=1.)
            complexity['{}:{}'.format(p, path)] = col

    # sc = {k: np.sqrt(v + np.log(1/delta_i)) for (k, v) in complexity.items()}
    # print('Sample complexity improvements:')
    # for k in programs:
    #     m_c = sc[k]
    #     d_c = sc['nighttime.t']
    #     print('{k}: {v:.2}x'.format(k=k, v=m_c/d_c))

    plot_theo = not args.no_theoretical
    plot_uniform = not args.no_uniform
    plot_optimal = not args.no_optimal
    plot_frequency = not args.no_frequency


    if args.print_dist:
        for (name, typ) in {'optimal': optimal_sampling,
                            'data': data_dist_sampling}.items():
            distr = typ(data_dist, complexity)
            print('{}: {} (n=1000: {})'.format(
                name,
                distr,
                {k: 1000 * v for (k, v) in distr.items()},
            ))

        # for p in weights:
        #     print('effective sample complexity for {}: {}'.format(p, np.sqrt(complexity[p] + np.log(1/delta_i))))
        return

    print('Computing theoretical errors...')
    print(data_dist)
    print(complexity)
    dist_errs = get_theoretical_and_empirical_errors(optimal_sampling, data_dist, complexity, no_train=args.no_train)
    uniform_errs = get_theoretical_and_empirical_errors(uniform_sampling, data_dist, complexity, no_train=args.no_train)
    test_errs = get_theoretical_and_empirical_errors(data_dist_sampling, data_dist, complexity, no_train=args.no_train)

    for (plname, (theo_idx, emp_idx)) in {
            'Surrogate': [1, 2],
    }.items():
        fig = plt.figure(figsize=(8, 6))
        plt.title('{} Error'.format(plname))

        ax = plt.gca()
        if plot_theo:
            ax2 = ax.twinx()

        if plot_optimal:
            mids, lows, ups = map(np.array, zip(*dist_errs[emp_idx]))
            m_xs = get_real_counts(optimal_sampling, data_dist, complexity)
            m_xs = mns
            ax.plot(m_xs, mids, '-', label='Optimal Sampling', color=C[0])
            ax.fill_between(m_xs, mids-lows, mids+ups, color=C[0], alpha=0.2)
            if plot_theo:
                ax2.plot(m_xs, dist_errs[theo_idx], label='Optimal Sampling', ls=':', color=C[0])

            frequency_mids = np.array([d[0] for d in test_errs[2]])
            imps = frequency_mids/mids
            print('geomean improvement over frequency: {}'.format(
                np.exp(np.mean(np.log(imps))) - 1
            ))
            theo_imps = np.array(test_errs[theo_idx])/np.array(dist_errs[theo_idx])
            print('geomean improvement over frequency (theoretical): {}'.format(
                np.exp(np.mean(np.log(theo_imps))) - 1
            ))

            uniform_mids = np.array([d[0] for d in uniform_errs[2]])
            imps = uniform_mids/mids
            print('geomean improvement over uniform: {}'.format(
                np.exp(np.mean(np.log(imps))) - 1
            ))
            theo_imps = np.array(uniform_errs[theo_idx])/np.array(dist_errs[theo_idx])
            print('geomean improvement over uniform (theoretical): {}'.format(
                np.exp(np.mean(np.log(theo_imps))) - 1
            ))
            # print('median improvement over frequency (where ns < 70): {}'.format(
            #     np.median(imps[mns < 70])
            # ))
            # print('median improvement over frequency (where ns >= 70): {}'.format(
            #     np.median(imps[mns >= 70])
            # ))

        if plot_frequency:
            mids, lows, ups = map(np.array, zip(*test_errs[emp_idx]))
            m_xs = get_real_counts(data_dist_sampling, data_dist, complexity)
            m_xs = mns
            ax.plot(m_xs, mids, '--', label='Frequency Sampling', color=C[2])
            ax.fill_between(m_xs, mids-lows, mids+ups, color=C[2], alpha=0.2)
            if plot_theo:
                ax2.plot(m_xs, test_errs[theo_idx], label='Frequency Sampling', ls=':', color=C[2])

        if plot_uniform:
            mids, lows, ups = map(np.array, zip(*uniform_errs[emp_idx]))
            m_xs = get_real_counts(uniform_sampling, data_dist, complexity)
            ax.plot(m_xs, mids, '-', label='Uniform Sampling', color=C[3])
            ax.fill_between(m_xs, mids-lows, mids+ups, color=C[3], alpha=0.2)
            if plot_theo:
                ax2.plot(m_xs, uniform_errs[theo_idx], label='Uniform Sampling', ls=':', color=C[3])


        ax.legend()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Total Dataset Size')
        ax.set_ylabel('Error')

        if plot_theo:
            ax2.set_yscale('log')
            # ax2.set_ylabel('Theoretical Error')

        fig.tight_layout()

        plot_utils.format_axes(ax)

    if not args.no_analysis:
        plt.figure()
        ax = plt.gca()
        ax2 = ax.twinx()
        for i, (n, es) in enumerate([
                ('dist', dist_errs),
                ('test', test_errs)
        ]):
            for (typ, fmt) in zip(es[-1][0].keys(), ['--', ':', '-.']):
                ax.plot(
                    mns,
                    [x[typ][0] for x in es[-1]],
                    label='{} {}'.format(n, typ),
                    color='C{}'.format(i),
                    ls=fmt,
                )

                ax2.plot(
                    mns,
                    [x[typ] for x in es[-2]],
                    label='{} {}'.format(n, typ),
                    color='C{}'.format(i),
                    ls=fmt,
                    alpha=0.5,
                )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax2.set_yscale('log')
        plt.legend()

    if True:
        plt.figure()
        ax = plt.gca()
        for i, (n, es) in enumerate([
                ('dist', dist_errs),
                ('test', test_errs)
        ]):
            for (typ, fmt) in zip(es[-1][0].keys(), ['--', ':', '-.']):
                ax.plot(
                    [x[typ] for x in es[-2]],
                    [x[typ][0] for x in es[-1]],
                    label='{} {}'.format(n, typ),
                    color='C{}'.format(i),
                    ls=fmt,
                )
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.legend()


    path_labels = {
        'nighttime.t': 'Nighttime',
        'twilight.t': 'Twilight',
        'daytime.t': 'Daytime',
    }


    if not args.no_per_path:
        fig = plt.figure(figsize=(8, 6))
        ax = plt.gca()

        if plot_theo:
            ax2 = ax.twinx()

        for idx, path in enumerate(sorted(data_dist.keys())[::-1]):
            data = []
            ogpath = path
            if ':' in path:
                path, ppath = path.split(':')
            else:
                ppath = 'all'
            for n in os.listdir('data/{}/{}'.format(path, ppath)):
                md = []
                if not os.path.exists('data/{}/{}/{}'.format(path, ppath, n)):
                    continue
                for t in os.listdir('data/{}/{}/{}'.format(path, ppath, n)):
                    md.append(torch.load('data/{}/{}/{}/{}/{}.loss'.format(path, ppath, n, t, 's' if ppath == 'all' else ppath)))
                md = np.array(md)
                data.append((int(n), np.mean(md), np.std(md) / np.sqrt(len(md))))
            xs, ys, errs = map(np.array, zip(*sorted(data)))

            print('path: {}, xs: {}, ys: {}, errs: {}'.format(path, xs, ys, errs))

            plt.plot(xs, ys, 'o--', label=path_labels.get(ogpath, ogpath), color=C[idx])
            plt.fill_between(xs, ys-errs * np.sqrt(N_TRIALS), ys+errs * np.sqrt(N_TRIALS), color=C[idx], alpha=0.2)

            if plot_theo:
                ax2.plot(xs, [get_theoretical_error(n, complexity[ogpath]) for n in xs], ls=':', color=C[idx])

        ax.set_xscale('log')
        ax.set_yscale('log')
        if plot_theo:
            ax2.set_yscale('log')

        ax.set_xlabel('Number of Path Samples')
        ax.set_ylabel('Error')
        plt.title('Sample Efficiencies of Paths')

        if plot_theo:
            ax2.get_yaxis().set_visible(False)

        fig.tight_layout()
        plt.legend()



    plt.show()

if __name__ == '__main__':
    main()
