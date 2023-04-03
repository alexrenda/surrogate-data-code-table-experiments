#!/usr/bin/env python

from typing import Any, Set, Optional, Union, Mapping, List, Tuple, Dict
import argparse
import collections
import numpy as np
import pandas as pd
import functools
import json
import os

from . import complexity
from . import interpret
from . import parser
from . import typecheck
from . import jacobian
from . import serialize
from .syntax import *
from .util import *

def calculate_complexity(p: Program, max_scaling: Union[float, Mapping[str, interpret.IValue]], verbose: bool = False) -> float:
    if isinstance(max_scaling, float):
        max_scaling_col = complexity.flat_scaling(p, max_scaling)
    else:
        max_scaling_col = max_scaling

    (a, b) = complexity.sum_complexities(complexity.complexity_interpret_program(p, max_scaling=complexity.flat_scaling(p, 0.)))
    (ap, bp) = complexity.sum_complexities(complexity.complexity_interpret_program(p, max_scaling=max_scaling_col))
    if verbose:
        print('Result: {}'.format((a, b)))

    return (bp + a)**2

def _parse_ivalue(x: str) -> interpret.IValue:
    return json.loads(x, parse_int=float) # type: ignore

def _parse_inputs(inputs: List[str]) -> Union[Mapping[str, interpret.IValue], float]:
    if any(':' in x for x in inputs):
        return {k: [_parse_ivalue(v)] for (k, v) in [x.split(':') for x in inputs]}
        # return {x.split(':')[0]: list(map(float, x.split(':')[1].split(','))) for x in inputs}
    elif len(inputs) == 1 and ',' not in inputs[0]:
        return float(inputs[0])
    else:
        raise NotImplementedError('Invalid input scaling: {}'.format(inputs))

def do_interpret(args: argparse.Namespace, program: Program) -> None:
    inputs = _parse_inputs(args.input)
    if isinstance(inputs, float):
        inputs = {k: [inputs] for k in program.inputs}

    # if len(program.inputs) != len(inputs):
    #     raise ValueError('Inputs must have length {}, got {} ({})'.format(len(program.inputs), len(inputs), program.inputs))
    print(interpret.interpret_program(program, inputs))


def do_complexity(args: argparse.Namespace, program: Program) -> None:
    max_scaling = _parse_inputs(args.input_scale)

    try:
        calc_complexity = calculate_complexity(program, max_scaling=max_scaling, verbose=args.verbose)
        print('full {}'.format(calc_complexity))
    except ValueError as e:
        pass

    all_paths = get_paths_of_program(program)
    for (linp, path) in all_paths:
        if args.paths and path not in args.paths:
            continue

        if args.partition:
            n = args.partition[0]
            all_partitions = get_partitions_of_program(linp, n)
            summary = {} # type: Dict[str, Dict[str, float]]

            col = calculate_complexity(linp, max_scaling=max_scaling, verbose=args.verbose)
            print('{} {}'.format(path, col))
            summary['full'] = {'program': col}

            # output partitions to files
            filename = os.path.basename(args.program).split('.')[0]
            folder_pre = os.path.join('tests','complexity', filename+'_partitions')

            for (i, (p_list, _)) in enumerate(all_partitions):
                partition_name = 'partitioning_' + str(i)
                summary[partition_name] = {}

                for j, p in enumerate(p_list):
                    folder = os.path.join(folder_pre, partition_name)

                    os.makedirs(folder, exist_ok=True)

                    with open(os.path.join(folder, 'program_{}.n'.format(j)), 'w') as f:
                        f.write(serialize.serialize_program(p))

                    # calculate complexity of each program in each paritioning
                    col = calculate_complexity(p, max_scaling=max_scaling, verbose=args.verbose)
                    print('{}-{}-{} {}'.format(path, i, j, col))

                    summary[partition_name]['program_{}'.format(j)] = col

                with open(os.path.join(folder_pre, 'summary.json'), 'w') as f:
                    json.dump(summary, f)

        else:
            col = calculate_complexity(linp, max_scaling=max_scaling, verbose=args.verbose)
            print('{} {}'.format(path, col))

def parse_float(x: str) -> float:
    return float(pd.to_numeric(x.replace('pi', str(np.pi)), errors='raise'))

def do_jacobian(args: argparse.Namespace, program: Program) -> None:
    if len(args.input_bounds) == 1:
        if ':' in args.input_bounds[0]:
            # evaluate floating point numbers in input, including performing division
            (lower, upper) = map(parse_float, args.input_bounds[0].split(':'))
            inputs = {k: _matrix_of_interval(v, jacobian.Interval(lower, upper)) for (k, v) in program.inputs.items()}
        else:
            raise NotImplementedError()
    else:
        inputs = {k: [jacobian.Interval(float(v.split(':')[0]), float(v.split(':')[1]))] for (k, v) in zip(program.inputs.keys(), args.input_bounds)}

    beta = float(max(_max_abs_interval_matrix(v) for v in inputs.values()))

    complexity, _, _ = get_complexity_of_program(program, beta, inputs)
    print('BETA: {}'.format(beta))
    print('complexity {}'.format(complexity))


    all_paths = get_paths_of_program(program)
    for (linp, path) in all_paths:
        if args.paths and path not in args.paths:
            continue

        complexity, _, jac = get_complexity_of_program(linp, beta, inputs)
        L = get_lipschitz_constant(jac, inputs, linp)
        print('{} {} {}'.format(path, complexity, L))


        print('Input bounds: {}'.format(inputs))
        linp = normalize_program_structure(linp)
        do_jacobian_trace(linp, beta, inputs)

def get_complexity_of_program(p: Program, beta: Union[float, Mapping[str, Matrix[float]]], inputs: Mapping[str, Matrix[jacobian.Interval]]) -> Tuple[float, Mapping[str, jacobian.JacVal], jacobian.ProgramJacobian]:
    if isinstance(beta, float):
        beta_col = complexity.flat_scaling(p, beta)
    else:
        beta_col = beta
    cl = np.sqrt(calculate_complexity(p, beta_col))
    jac_state, result_jac = jacobian.jacobian_interpret_program(p, inputs)

    return cl, jac_state, result_jac

def get_lipschitz_constant(jac: List[List[jacobian.Interval]], inputs: Mapping[str, Matrix[jacobian.Interval]], p: Program) -> float:
    jac_array = np.array([[[x.l, x.r] for x in v] for v in jac])
    jac_max = np.abs(jac_array).max(axis=2)
    jac_cols = jac_max.sum(axis=1)

    # ALEX NOTE: I don't know what this was doing, but it looks wrong
    # legal_idxs = []
    # i = 0
    # for k in inputs:
    #     for v in range(len(inputs[k])):
    #         # check if VectorAccess(k, v) is in l's outputs
    #         assert isinstance(p.output, Vector)
    #         if VectorAccess(Variable(k), v) in p.output.exprs:
    #             legal_idxs.append(i)
    #         i += 1
    # jac_cols = jac_cols[legal_idxs]

    L = jac_cols.max()
    return float(L)

def get_complexity_of_split(l: Program, r: Program, inputs: Mapping[str, Matrix[jacobian.Interval]], beta: float) -> Tuple[float, float, float, float]:
    print('-'*80)
    print(l)
    print(r)
    print('-'*80)

    left_complexity, left_jac_state, _ = get_complexity_of_program(l, beta, inputs)
    new_inputs = {k: jacobian.dual_interval_to_interval(v, 'l') for (k, v) in left_jac_state.items() if k in r.inputs}

    def rebound_inputs(x: Matrix[jacobian.Interval]) -> Matrix[jacobian.Interval]:
        if isinstance(x, jacobian.Interval):
            return jacobian.Interval(-abs(x.l), abs(x.l))
        else:
            return [rebound_inputs(v) for v in x]

    new_inputs = {k: rebound_inputs(v) for (k, v) in new_inputs.items()}

    r_beta = {k: jacobian.interval_to_float(v, 'r') for (k, v) in new_inputs.items()}

    right_complexity, _, result_jac = get_complexity_of_program(r, r_beta, new_inputs)

    # 3d array. Rows are inputs, columns are outputs, depth is interval
    L = get_lipschitz_constant(result_jac, new_inputs, l)

    return left_complexity, right_complexity, L, L*left_complexity+right_complexity

def print_program(idx: int, p: Tuple[Optional[Program], Optional[Program]]) -> None:
    fstr = ' SPLIT IDX: {} '.format(idx)
    eq_len = (80 - len(fstr))//2
    (l, r) = p
    print('='*eq_len + fstr + '='*eq_len)
    if l is not None:
        print(serialize.serialize_program(l))
    print('*' * 80)
    if r is not None:
        print(serialize.serialize_program(r))
    print('*' * 81)

def print_complexity(left: float, right: float, L: float, total: float) -> None:
    print('Left complexity: {}'.format(left))
    print('Right complexity: {}'.format(right))
    print('Right lipschitz: {}'.format(L))
    print('Composition complexity: {}'.format(total))


def _matrix_of_interval(x: List[int], i: T) -> Matrix[T]:
    if x:
        return [ _matrix_of_interval(x[1:], i) for _ in range(x[0]) ]
    else:
        return i

def _max_abs_interval_matrix(x: Matrix[jacobian.Interval]) -> float:
    if isinstance(x, list):
        return max(_max_abs_interval_matrix(v) for v in x)
    else:
        return max(abs(x.l), abs(x.r))

def do_jacobian_trace(program: Program, beta: float, inputs: Mapping[str, Matrix[jacobian.Interval]]) -> None:
    # original_complexity = np.sqrt(calculate_complexity(program, beta))
    # print('Original complexity: {}'.format(original_complexity))

    best_complexity  = np.inf
    best_idx = None
    best_split = None
    best_col = None

    # print('Original complexity: {}'.format(original_complexity))


    # Calculate the complexity of the split programs
    splits = get_binary_splits(program)
    for idx, (l, r) in enumerate(splits):
        if l.statement == Skip() or r.statement == Skip() or l.statement is None or r.statement is None:
            continue

        left_complexity, right_complexity, L, total_complexity = get_complexity_of_split(l, r, inputs, beta)

        if total_complexity < best_complexity:
            best_complexity = total_complexity
            best_idx = idx
            best_split = (l, r)
            best_col = (left_complexity, right_complexity, L, total_complexity)

        print_program(idx, (l, r))
        print_complexity(left_complexity, right_complexity, L, total_complexity)

        # if total_complexity < original_complexity:
        #     print_program(idx, (l, r))
        #     print_complexity(left_complexity, right_complexity, L, total_complexity)

    # if best_complexity < original_complexity:
    #     print_program(best_idx, best_split)
    #     m_col = get_complexity_of_split(*best_split, inputs, beta)
    #     assert np.allclose(m_col, best_col)
    #     print_complexity(*best_col)

    #     print('======\nComplexity Ratio: {}'.format(best_complexity/original_complexity))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--no-dce', action='store_true')
    p.add_argument('--program', required=True)
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--paths', nargs='+')
    # subparsers: standard interpretation, complexity interpretation, jacobian interpretation
    sp = p.add_subparsers(dest='subparser_name')
    ip = sp.add_parser('interpret')
    ip.add_argument('--input', required=True, nargs='*')
    cp = sp.add_parser('complexity')
    cp.add_argument('--input-scale', required=True, nargs='*')
    cp.add_argument('--partition', nargs=1, type=int)
    jp = sp.add_parser('jacobian')
    jp.add_argument('--input-bounds', required=True, nargs='*')
    args = p.parse_args()

    with open(args.program) as f:
        program = parser.parse_program(f.read())

    if args.verbose:
        print(serialize.serialize_program(program))

    if not args.no_dce:
        program = dce(normalize_program_structure(dce(program)))

    if args.verbose:
        print(serialize.serialize_program(program))

    if typecheck.typecheck_program(program) is None:
        print('Does not typecheck')
        return

    if args.subparser_name == 'interpret':
        do_interpret(args, program)
    elif args.subparser_name == 'complexity':
        do_complexity(args, program)
    elif args.subparser_name == 'jacobian':
        print(program)
        do_jacobian(args, program)
    else:
        raise ValueError('Unknown subparser name: {}'.format(args.subparser_name))


if __name__ == '__main__':
    main()
