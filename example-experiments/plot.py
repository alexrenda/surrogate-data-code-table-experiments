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
import multiprocessing.dummy as mp

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
    (213, 94, 0),
    (204, 121, 167),
]) / 255

LS = ['-', '--', ':', '-.']
MS = ['o', 's', 'v', 'D', 'P', 'X', 'd', 'p', 'h', 'H', '8', '4', '3', '2', '1', 'x', '+', 'd', 'p', 'h', 'H', '8', '4', '3', '2', '1', 'x', '+']

_PYTHON = sys.executable
CURR_DIR = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
PAR_DIR = os.path.dirname(CURR_DIR)

delta = 0.1

N_TRIALS = 5

mns = list(np.arange(1, 20)) + list(np.logspace(1, 5, 10).round().astype(int))
mns = np.logspace(1, 3, 10)
if any('jmeint' in s for s in sys.argv):
    mns = np.logspace(3, 5, 10)

mns = sorted(np.array(list(mns)).round().astype(int))
mns = np.array(mns)

def get_theoretical_error(n_samples, complexity, c):
    # n_samples = int(np.ceil(n_samples))
    C = 1
    #  - np.log(1 - (1 - delta)**(1/c))
    return np.sqrt(C * (complexity) / n_samples)

def get_empirical_error(prog_name_base, n_samples, no_train=False):
    losses = []
    fnames = []
    procs = []


    if ':' in prog_name_base:
        prog_name_base, path_name = prog_name_base.split(':')
    else:
        path_name = 'all'

    if 'jmeint' in prog_name_base and path_name == 'lllrrrrrlrrl' and n_samples == 35:
        n_samples = 26
    elif 'jmeint' in prog_name_base and path_name == 'lllrrrrlrrl' and n_samples == 35:
        n_samples = 26

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

    if len(losses) == 0:
        return None

    # losses.pop(np.argmax(losses))

    import scipy.stats
    mid = np.mean(losses)
    lower = np.std(losses) / np.sqrt(len(losses))
    upper = np.std(losses) / np.sqrt(len(losses))

    if False:
        dname = 'data/{prog_name_base}/{path_name}'.format(prog_name_base=prog_name_base, n=n_samples,trial=i, path_name=path_name, lname=path_name if path_name != 'all' else 's')
        ns = os.listdir(dname)
        ns = [int(n) for n in ns if n.isdigit()]
        ns = sorted(ns)

        xs = []
        ys = []

        for n in ns:
            d2name = 'data/{prog_name_base}/{path_name}/{n}'.format(prog_name_base=prog_name_base, n=n, path_name=path_name)
            trials = os.listdir(d2name)
            trials = [int(t) for t in trials if t.isdigit()]
            trials = sorted(trials)

            for t in trials:
                fname = 'data/{prog_name_base}/{path_name}/{n}/{trial}/{lname}.loss'.format(prog_name_base=prog_name_base, n=n,trial=t, path_name=path_name, lname=path_name if path_name != 'all' else 's')
                loss = torch.load(fname)
                xs.append(n)
                ys.append(loss)

        import scipy.stats
        # xs, ys show are linear on a log-log plot
        xs = np.array(xs)
        ys = np.array(ys)

        # savitzky_golay
        # yhat = scipy.signal.savgol_filter(ys, 50, 3)

        median_filter_size = 51

        # left and right pad with the first and last values
        x_pad = np.concatenate([np.ones(median_filter_size//2) * xs[0], xs, np.ones(median_filter_size//2) * xs[-1]])
        y_pad = np.concatenate([np.ones(median_filter_size//2) * ys[0], ys, np.ones(median_filter_size//2) * ys[-1]])

        # median filter
        yhat = np.array([np.median(y_pad[i:i+median_filter_size]) for i in range(len(y_pad) - median_filter_size + 1)])

        target_x = n_samples
        x_idx = np.argmin(np.abs(xs - target_x))
        target_y = ys[x_idx]
        mid = target_y


        # # interpolate to get value at n_samples
        # f = scipy.interpolate.interp1d(xs, yhat, kind='cubic')
        # mid = f(n_samples)
        # return mid, 0, 0



        # # curve fit
        # def myExpFunc(x, a, b):
        #     return a * np.power(x, b)

        # popt, pcov = scipy.optimize.curve_fit(myExpFunc, xs, ys)

        # # plot
        # plt.plot(xs, ys, 'o', label='data')
        # plt.plot(xs, myExpFunc(xs, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
        # plt.xlabel('sample size')
        # plt.ylabel('loss')

        # plt.xscale('log')
        # plt.yscale('log')

        # plt.legend()
        # plt.show()

        # import sys; sys.exit(1)


        # return y, err, err


    return mid, lower, upper

def optimal_sampling(data_distribution, complexity):
    n = len(data_distribution)
    c = len(data_distribution)
    weights = {
        k: (data_distribution[k] * np.sqrt(complexity[k] - np.log(1 - (1 - delta)**(1/c))))**(2/3)
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
        k: get_theoretical_error(v * n, complexities[k], len(per_stratum_prob))
        for (k, v) in per_stratum_prob.items()
    } for n in mns]

    per_stratum_empirical_error = []
    with mp.Pool(3) as pool:
        for n in tqdm.tqdm(mns):
            tasks = [(k, int(np.ceil(v * n))) for (k, v) in per_stratum_prob.items()]
            per_stratum_empirical_error.append({
                task[0]: res
                for (task, res) in zip(tasks, pool.starmap(get_empirical_error, tasks))
            })

    # {
    #     k: get_empirical_error(k, int(np.ceil(v * n)), no_train=no_train)
    #     for (k, v) in per_stratum_prob.items()
    # } for n in tqdm.tqdm(mns)]

    theoretical_expectation_error = [
        sum(data_dist[k] * v for (k, v) in d.items())
        for d in per_stratum_theoretical_error
    ]

    empirical_expectation_error = [
        (
            sum(data_dist[k] * v[0] for (k, v) in d.items() if v is not None),
            sum(data_dist[k] * v[1] for (k, v) in d.items() if v is not None),
            sum(data_dist[k] * v[2] for (k, v) in d.items() if v is not None),
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
    parser.add_argument('--print-table', action='store_true')
    parser.add_argument('--no-train', action='store_true')
    parser.add_argument('--no-uniform', action='store_true')
    parser.add_argument('--no-theoretical', action='store_true')
    parser.add_argument('--no-optimal', action='store_true')
    parser.add_argument('--no-frequency', action='store_true')
    parser.add_argument('--no-analysis', action='store_true')
    parser.add_argument('--no-per-path', action='store_true')
    parser.add_argument('--only-per-path', action='store_true')
    parser.add_argument('--no-show', action='store_true')
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

        if os.path.exists('datasets/{p}'.format(p=p)) or args.print_dist:
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
            if path not in program.config.distribution:
                continue
            col = turaco.main.calculate_complexity(linp, max_scaling=1.)
            complexity['{}:{}'.format(p, path)] = col

    compl_dist = optimal_sampling(data_dist, complexity)

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
        print('Complexity : {}'.format(complexity))

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

    if not args.only_per_path:
    # print('Computing theoretical errors...')
    # print(data_dist)
    # print(complexity)
        dist_errs = get_theoretical_and_empirical_errors(optimal_sampling, data_dist, complexity, no_train=args.no_train)
        if not args.no_uniform:
            uniform_errs = get_theoretical_and_empirical_errors(uniform_sampling, data_dist, complexity, no_train=args.no_train)
        if not args.no_frequency:
            test_errs = get_theoretical_and_empirical_errors(data_dist_sampling, data_dist, complexity, no_train=args.no_train)

    if args.print_table:
        theo_idx = 1
        emp_idx = 2
        mids, lows, ups = map(np.array, zip(*dist_errs[emp_idx]))
        m_xs = mns

        # print(r'\textbf{Path} & \textbf{Frequency} & \textbf{Complexity} & \textbf{Complexity-Guided Sampling} \\ \midrule')

        # frequency_mids = np.array([d[0] for d in test_errs[2]])
        # imps = frequency_mids/mids
        # print('geomean improvement over frequency: {}'.format(
        #     np.exp(np.mean(np.log(imps))) - 1
        # ))
        # theo_imps = np.array(test_errs[theo_idx])/np.array(dist_errs[theo_idx])
        # print('geomean improvement over frequency (theoretical): {}'.format(
        #     np.exp(np.mean(np.log(theo_imps))) - 1
        # ))

        # uniform_mids = np.array([d[0] for d in uniform_errs[2]])
        # imps = uniform_mids/mids
        # print('geomean improvement over uniform: {}'.format(
        #     np.exp(np.mean(np.log(imps)[1:])) - 1
        # ))
        # theo_imps = np.array(uniform_errs[theo_idx])/np.array(dist_errs[theo_idx])
        # print('geomean improvement over uniform (theoretical): {}'.format(
        #     np.exp(np.mean(np.log(theo_imps))) - 1
        # ))

        for (i, pth) in enumerate(sorted(data_dist.keys())):
            print('{pth} & {comp:.2f} & ${freq:.2%}$ & ${unif:.2%}$ & ${cdist:.2%}$  \\\\'.format(
                pth=pth.split(':')[1],
                comp=complexity[pth],
                freq=data_dist[pth],
                unif=1/len(data_dist),
                cdist=compl_dist[pth],
            ).replace('%', r'\%'))

        return


    if not args.only_per_path:
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

                if not args.no_frequency:
                    frequency_mids = np.array([d[0] for d in test_errs[2]])
                    imps = frequency_mids/mids
                    print('geomean improvement over frequency: {}'.format(
                        np.exp(np.mean(np.log(imps))) - 1
                    ))
                    theo_imps = np.array(test_errs[theo_idx])/np.array(dist_errs[theo_idx])
                    print('geomean improvement over frequency (theoretical): {}'.format(
                        np.exp(np.mean(np.log(theo_imps))) - 1
                    ))

                if not args.no_uniform:
                    uniform_mids = np.array([d[0] for d in uniform_errs[2]])
                    imps = uniform_mids/mids
                    print('geomean improvement over uniform: {}'.format(
                        np.exp(np.mean(np.log(imps)[1:])) - 1
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

            if plot_frequency and not args.no_show:
                mids, lows, ups = map(np.array, zip(*test_errs[emp_idx]))
                m_xs = get_real_counts(data_dist_sampling, data_dist, complexity)
                m_xs = mns
                ax.plot(m_xs, mids, '--', label='Frequency Sampling', color=C[2])
                ax.fill_between(m_xs, mids-lows, mids+ups, color=C[2], alpha=0.2)
                if plot_theo:
                    ax2.plot(m_xs, test_errs[theo_idx], label='Frequency Sampling', ls=':', color=C[2])

            if plot_uniform and not args.no_show:
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

        if not args.no_analysis and not args.no_show:
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
                        [x[typ][0] for x in es[-1] if x is not None and x[typ] is not None],
                        label='{} {}'.format(n, typ),
                        color='C{}'.format(i),
                        ls=fmt,
                        marker='o',
                    )

                    ax2.plot(
                        mns,
                        [x[typ] for x in es[-2]],
                        label='{} {}'.format(n, typ),
                        color='C{}'.format(i),
                        ls=fmt,
                        alpha=0.5,
                        marker='o',
                    )

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax2.set_yscale('log')
            plt.legend()

        import itertools
        sts = list(itertools.product(C, LS))
        import random
        random.Random(0).shuffle(sts)
        sts = itertools.cycle(sts)

        if not args.no_show:
            plt.figure()
            ax = plt.gca()

            for typ in dist_errs[-1][0].keys():
               c, ls = next(sts)
               mms = itertools.cycle(MS)
               for i, (n, es) in enumerate([
                        ('dist', dist_errs),
                        ('test', test_errs)
               ]):
                   ax.plot(
                       [x[typ] for x in es[-2] if x is not None and x[typ] is not None],
                       [x[typ][0] for x in es[-1] if x is not None and x[typ] is not None],
                       label='{} {}'.format(n, typ),
                       color=c,
                       ls=ls,
                       marker=next(mms),
                   )


            # for i, (n, es) in enumerate([
            #         ('dist', dist_errs),
            #         ('test', test_errs),
            #         ('uniform', uniform_errs),
            # ]):
            #     ls = LS[i]
            #     idx = 0
            #     for (typ, m) in zip(es[-1][0].keys(), MS):
            #         c = sts[idx][0]
            #         ax.plot(
            #             [x[typ] for x in es[-2]],
            #             [x[typ][0] for x in es[-1]],
            #             label='{} {}'.format(n, typ),
            #             color=c,
            #             ls=ls,
            #             marker=m,
            #         )
            #         idx += 1
            ax.set_xscale('log')
            # ax.set_yscale('log')
            plt.legend()


    path_labels = {
        'nighttime.t': 'Nighttime',
        'twilight.t': 'Twilight',
        'daytime.t': 'Daytime',
    }

    import itertools
    sts = list(itertools.product(C, LS))
    import random
    random.Random(0).shuffle(sts)


    if not args.no_per_path and not args.no_show:
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

            plt.plot(xs, ys, 'o--', label=path_labels.get(ogpath, ogpath), color=sts[idx][0], ls=sts[idx][1])
            plt.fill_between(xs, ys-errs * np.sqrt(N_TRIALS), ys+errs * np.sqrt(N_TRIALS), alpha=0.2, color=sts[idx][0])

            # if plot_theo:
            #     ax2.plot(xs, [get_theoretical_error(n, complexity[ogpath]) for n in xs], ls=':', color=C[idx])

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



    if not args.no_show:
        plt.show()

if __name__ == '__main__':
    main()
