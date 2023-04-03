from typing import Any, Dict, Set, Optional, Union, Mapping, List, Tuple, TypeVar, Generic, Callable, Iterator
from .syntax import *
from . import typecheck
import itertools
from pprint import pprint
import collections
import base64
import pickle

T = TypeVar('T')
S = TypeVar('S')
# define Matrix to be a recursive type parameterized by a type T
Matrix = Union[T, List['Matrix[T]']]

def _apply_binary_operator(l: Matrix[T], r: Matrix[T], op: Callable[[T, T], S]) -> Matrix[S]:
    if isinstance(l, list) and isinstance(r, list):
        return [_apply_binary_operator(x, y, op) for (x, y) in zip(l, r)]
    elif isinstance(l, list):
        return [_apply_binary_operator(x, r, op) for x in l]
    elif isinstance(r, list):
        return [_apply_binary_operator(l, y, op) for y in r]
    else:
        return op(l, r)

def _apply_unary_operator(v: Matrix[T], op: Callable[[T], S]) -> Matrix[S]:
    if isinstance(v, list):
        return [_apply_unary_operator(x, op) for x in v]
    else:
        return op(v)

def _assert_predicate(v: Matrix[T], p: Callable[[T], bool]) -> None:
    if isinstance(v, list):
        for x in v:
            _assert_predicate(x, p)
    else:
        assert p(v)

def _reduce(v: Matrix[T], op: Callable[[T, T], T]) -> T:
    if isinstance(v, list):
        reduced = [_reduce(x, op) for x in v] # type: List[T]
        z = reduced[0]
        for x in reduced[1:]:
            z = op(z, x)
        return z
    else:
        return v

def _updated_variables(statement: Statement) -> Set[str]:
    if isinstance(statement, Sequence):
        return _updated_variables(statement.left) | _updated_variables(statement.right)
    elif isinstance(statement, IfThen):
        return _updated_variables(statement.left) | _updated_variables(statement.right)
    # elif isinstance(statement, Repeat):
    #     return _updated_variables(statement.s)
    elif isinstance(statement, Assignment):
        return {statement.name}
    elif isinstance(statement, Skip):
        return set()
    elif isinstance(statement, Print):
        return set()
    else:
        raise ValueError(statement)


def get_paths_of_program(program: Program) -> List[Tuple[Program, str]]:
    def _has_branch(statement: Statement) -> bool:
        if isinstance(statement, Sequence):
            return _has_branch(statement.left) or _has_branch(statement.right)
        elif isinstance(statement, IfThen):
            return True
        # elif isinstance(statement, Repeat):
        #     return _has_branch(statement.s)
        else:
            return False

    def _get_paths_from_statement(statement: Statement) -> List[Tuple[Statement, str]]:
        if isinstance(statement, Sequence):
            left_paths = _get_paths_from_statement(statement.left)
            right_paths = _get_paths_from_statement(statement.right)
            return [
                (Sequence(ls, rs), lp+rp) for ((ls, lp), (rs, rp)) in itertools.product(left_paths, right_paths)
            ]
        elif isinstance(statement, IfThen):
            return ([(s, 'l'+p) for (s, p) in _get_paths_from_statement(statement.left)] +
                    [(s, 'r'+p) for (s, p) in _get_paths_from_statement(statement.right)])
        # elif isinstance(statement, Repeat):
        #     if not _has_branch(statement.s):
        #         return [(statement, '')]

        #     if statement.n == 0:
        #         return [(statement, '')]
        #     else:
        #         return _get_paths_from_statement(Sequence(statement.s, Repeat(statement.n-1, statement.s)))
        elif isinstance(statement, Assignment):
            return [(statement, '')]
        elif isinstance(statement, Skip):
            return [(statement, '')]
        elif isinstance(statement, Print):
            return [(statement, '')]
        elif isinstance(statement, Iterate):
            return [(statement, '')]
        else:
            raise ValueError(statement)

    return [
        (Program(program.inputs, statement, program.outputs), path)
        for (statement, path) in _get_paths_from_statement(program.statement)
    ]

def _get_vars(p: Program) -> Dict[str, Statement]:
    env = {} # type: Dict[str, Statement]

    def get_vars_statement(s: Statement) -> None:
        if isinstance(s, Assignment):
            # Ignore indices, just treat the assignment to the whole variable
            env[s.name] = s
        elif isinstance(s, Sequence):
            get_vars_statement(s.left)
            get_vars_statement(s.right)
        elif isinstance(s, IfThen):
            raise NotImplementedError()
        elif isinstance(s, Iterate):
            raise NotImplementedError()
        elif isinstance(s, Print):
            raise NotImplementedError()
        elif isinstance(s, Skip):
            pass
        else:
            raise ValueError(s)

    get_vars_statement(p.statement)
    return env

def _expr_depends_on(e: Expr) -> List[str]:
    if isinstance(e, Variable):
        return [e.name]
    elif isinstance(e, Value):
        return []
    elif isinstance(e, Binop):
        return _expr_depends_on(e.left) + _expr_depends_on(e.right)
    elif isinstance(e, Unop):
        return _expr_depends_on(e.expr)
    elif isinstance(e, Vector):
        return list(itertools.chain(*[_expr_depends_on(x) for x in e.exprs]))
    elif isinstance(e, VectorAccess):
        return _expr_depends_on(e.expr) + ([e.index] if isinstance(e.index, str) else [])
    else:
        raise ValueError(e)

def _statement_depends_on(s: Statement) -> List[str]:
    if isinstance(s, Assignment):
        return _expr_depends_on(s.value)
        # + [x for x in s.indices if isinstance(x, str)]
    elif isinstance(s, Sequence):
        # shouldn't ever get a sequence here
        raise ValueError(s)
    elif isinstance(s, IfThen):
        raise NotImplementedError()
    elif isinstance(s, Iterate):
        raise NotImplementedError()
        # return _statement_depends_on(s.s) \
        #     + ([s.start] if isinstance(s.start, str) else []) \
        #     + ([s.end] if isinstance(s.end, str) else [])
    elif isinstance(s, Print):
        raise NotImplementedError()
    elif isinstance(s, Skip):
        return []
    else:
        raise ValueError(s)

# Check partitions are valid (no cyclic dependencies) and topological sort partitions
def _topo_sort_partition(env: Dict[str, Statement], partition: List[Set[str]]) -> Optional[List[Set[str]]]:

    id2partition = {i: partition[i] for i in range(len(partition))}
    var2partitionid = {v: i for (i, p) in id2partition.items() for v in p}

    sorted_partition = []

    # topo sort
    def _topo_sort() -> None:
        VISITED = 2
        VISITING = 1
        UNVISITED = 0
        status = {id:UNVISITED for id in id2partition.keys()}
        unvisited_ids = set(id2partition.keys())

        def _visit(id: int) -> None:
            if status[id] == VISITED:
                return
            if status[id] == VISITING:
                raise ValueError("Cycle")

            status[id] = VISITING
            unvisited_ids.remove(id)

            to_visit = set()
            for v in id2partition[id]:
                # assume all undefined variables are inputs
                if v in env:
                    for d in set(_statement_depends_on(env[v])):
                        # assume all undefined variables are inputs
                        if d in var2partitionid:
                            dep_part_id = var2partitionid[d]
                            # if in the same partition, doesn't matter
                            if dep_part_id != id:
                                to_visit.add(var2partitionid[d])

            for j in to_visit:
                _visit(j)

            status[id] = VISITED
            sorted_partition.append(id2partition[id])

        while len(unvisited_ids) > 0:
            id = unvisited_ids.pop()
            unvisited_ids.add(id)
            _visit(id)

    try:
        _topo_sort()
        return sorted_partition
    except ValueError:
        return None

def size_of_variables(p: Program) -> Dict[str, List[int]]:
    sizes = {v:s for (v, s) in p.inputs.items()}

    ctx = typecheck.get_program_types(p)

    # ctx is never None because we typechecked the program before
    assert ctx is not None

    for var, typ in ctx.vars.items():
        if var not in sizes:
            sizes[var] = typ.to_list()

    return sizes

# assumes output is just a variable (no iterates, no ite)
# would probably just need to create "dummy" variable for iterates
def _construct_programs(p: Program, env: Dict[str, Statement], partition: List[Set[str]]) -> List[Program]:
    # get size of all_vars
    sizes = size_of_variables(p)

    programs = []

    prev_undefined_vars = set()
    for output in p.outputs:
        if isinstance(output, Variable):
            prev_undefined_vars.add(output.name)
        else:
            raise ValueError(output)

    def _construct_statement(vars: Set[str]) -> Tuple[Optional[Statement], Set[str]]:
        curr_vars = set() # type: Set[str]
        curr_undefined_vars = set() # type: Set[str]

        visited = set() # type: Set[str]

        def _visit(v: str, curr_statement: Optional[Statement]) -> Optional[Statement]:
            if v in visited:
                return curr_statement
            visited.add(v)

            if v in env:
                for d in set(_statement_depends_on(env[v])):
                    # assume all undefined variables are inputs
                    if d in part:
                        curr_statement = _visit(d, curr_statement)
                    else:
                        curr_undefined_vars.add(d)

            if v in part:
                if curr_statement is None:
                    curr_statement = env[v]
                else:
                    curr_statement = Sequence(curr_statement, env[v])

                curr_vars.add(v)
            else:
                raise ValueError(v)

            return curr_statement

        curr_statement = None
        for v in vars:
            curr_statement = _visit(v, curr_statement)

        return curr_statement, curr_undefined_vars

    for part in reversed(partition):
        curr_statement, curr_undefined_vars = _construct_statement(part)


        # for v in part:
        #     if v in env:
        #         if curr_statement is None:
        #             curr_statement = env[v]
        #         else:
        #             curr_statement = Sequence(curr_statement, env[v])

        #         curr_vars.add(v)

        if curr_statement is None:
            raise ValueError(part)

        curr_outputs = [] # type: List[Expr]
        defined = set() # type: Set[str]
        for output2 in prev_undefined_vars:
            if output2 in part:
                curr_outputs.append(Variable(output2))
                defined.add(output2)

        prev_undefined_vars -= defined

        # curr_undefined_vars = set()
        # for v in curr_vars:
        #     curr_undefined_vars = curr_undefined_vars.union(
        #         set(_statement_depends_on(env[v]))
        #     )

        # curr_undefined_vars -= curr_vars

        curr_inputs = {}
        for v in curr_undefined_vars:
            if v in sizes:
                curr_inputs[v] = sizes[v]
            else:
                raise ValueError(v)

        prev_undefined_vars = prev_undefined_vars.union(curr_undefined_vars)

        programs.append(
            Program(
                inputs=curr_inputs,
                statement=curr_statement,
                outputs=curr_outputs,
            )
        )

    return programs

# Brute force through all possible (non-overlapping) partitions
# of the program up to n sub-programs
# Assumes all undefined variables are inputs
# Returns list of tuples
# (list of sub-programs, base64 encoding of vars in each sub-program)
def get_partitions_of_program(p: Program, n: int) -> List[Tuple[List[Program], str]]:
    if n == 1:
        return [([p], '')]

    env = _get_vars(p)
    vars = list(env.keys())
    # vars = ['out1a', 'skyLight', 'emission', 'ambient', 'groundLight']

    print('n vars:', len(vars))

    def get_partitions(unused: Set[str], i_partition: int, partitions: List[Set[str]]) -> Iterator[List[Set[str]]]:
        if i_partition == n:
            yield partitions + [unused]
        else:
            for size in range(1, len(unused) - (n - i_partition) + 1):
                for partition in itertools.combinations(unused, size):
                    new_unused = unused - set(partition)
                    new_partitions = partitions + [set(partition)]
                    if len(new_unused) == 0:
                        continue
                    yield from get_partitions(new_unused, i_partition+1, new_partitions)

    # Construct programs from partitions
    partitions = get_partitions(set(vars), 1, [])

    n_total = 0
    n_good = 0

    program_partitions = []

    for partition in partitions:
        n_total += 1

        sorted_partition = _topo_sort_partition(env, partition)

        if sorted_partition is None:
            continue

        ident = base64.b64encode(pickle.dumps(sorted_partition)).decode('utf-8')

        programs = _construct_programs(p, env, sorted_partition)
        program_partitions.append((programs, ident))

        n_good += 1

    print('n partitions:', n_total)
    print('n good partitions:', n_good)

    return program_partitions

def skip_remove(s: Statement) -> Statement:
    if isinstance(s, Sequence):
        l = skip_remove(s.left)
        r = skip_remove(s.right)
        if isinstance(l, Skip):
            return r
        elif isinstance(r, Skip):
            return l
        else:
            return Sequence(l, r)
    return s

def dce(p: Program) -> Program:
    def _variables_of_expr(e: Expr) -> Set[str]:
        if isinstance(e, Variable):
            return {e.name}
        elif isinstance(e, Value):
            return set()
        elif isinstance(e, Binop):
            return _variables_of_expr(e.left) | _variables_of_expr(e.right)
        elif isinstance(e, Unop) or isinstance(e, ConstrainedUnop):
            return _variables_of_expr(e.expr)
        elif isinstance(e, Vector):
            return set(x for y in e.exprs for x in _variables_of_expr(y))
        elif isinstance(e, VectorAccess):
            return _variables_of_expr(e.expr)
        else:
            raise ValueError(e)

    def _dce(s: Statement, l: Set[str]) -> Tuple[Statement, Set[str]]:
        # Sequence, IfThen, Repeat, Assignment, Skip, Print
        if isinstance(s, Sequence):
            rp, l = _dce(s.right, l)
            lp, l = _dce(s.left, l)
            return Sequence(lp, rp), l
        elif isinstance(s, IfThen):
            lp, ll = _dce(s.left, l)
            rp, rl = _dce(s.right, l)
            l = ll | rl | _variables_of_expr(s.condition)
            return IfThen(s.condition, lp, rp), l
        elif isinstance(s, Assignment):
            if s.name not in l:
                print('DCE: Removing assignment to', s.name)
                return Skip(), l

            live = _variables_of_expr(s.value)
            if s.indices:
                live |= set(y for y in s.indices if isinstance(y, str))

            l = l - {s.name} | live
            return s, l
        elif isinstance(s, Skip):
            return Skip(), l
        elif isinstance(s, Print):
            live = _variables_of_expr(s.value)
            l = l | live
            return Print(s.value), l
            raise NotImplementedError()
        elif isinstance(s, Iterate):
            raise NotImplementedError()
        else:
            raise ValueError(s)

    live = set(y for x in p.outputs for y in _variables_of_expr(x))
    prog, _ = _dce(p.statement, live)
    prog = skip_remove(prog)
    return Program(p.inputs, prog, p.outputs)

def normalize_program_structure(p: Program) -> Program:
    # make it so that the program is right-recursive in sequences
    def _normalize(s: Statement) -> Statement:
        if isinstance(s, Sequence):
            if isinstance(s.left, Sequence):
                new_left = _normalize(s.left)
                assert isinstance(new_left, Sequence)
                new_right = Sequence(new_left.right, s.right)
                return Sequence(new_left.left, _normalize(new_right))
            else:
                return Sequence(s.left, _normalize(s.right))
        elif isinstance(s, IfThen):
            return IfThen(s.condition, _normalize(s.left), _normalize(s.right))
        # elif isinstance(s, Repeat):
        #     return Repeat(s.n, _normalize(s.s))
        elif isinstance(s, Assignment):
            return s
        elif isinstance(s, Skip):
            return Skip()
        elif isinstance(s, Print):
            return Print(s.value)
        elif isinstance(s, Iterate):
            return s
        else:
            raise ValueError(s)

    # remove skips
    def _remove_skips(s: Statement) -> Statement:
        if isinstance(s, Sequence):
            if isinstance(s.left, Skip):
                return _remove_skips(s.right)
            elif isinstance(s.right, Skip):
                return _remove_skips(s.left)
            else:
                return Sequence(_remove_skips(s.left), _remove_skips(s.right))
        elif isinstance(s, IfThen):
            return IfThen(s.condition, _remove_skips(s.left), _remove_skips(s.right))
        # elif isinstance(s, Repeat):
        #     return Repeat(s.n, _remove_skips(s.s))
        elif isinstance(s, Assignment):
            return s
        elif isinstance(s, Skip):
            return Skip()
        elif isinstance(s, Print):
            return Print(s.value)
        elif isinstance(s, Iterate):
            return s
        else:
            raise ValueError(s)

    return Program(p.inputs, _remove_skips(_normalize(p.statement)), p.outputs)

# def get_binary_splits(p: Program) -> List[Tuple[Program, Program]]:
#     # iterate through the program, splitting at each line
#     # don't split underneath if statements or repeat statements
#     # assume that the program is normalized

#     st = p.statement
#     assert isinstance(st, Sequence), 'Program must be a sequence'

#     # create splits
#     suffix = st.right
#     res = [(st.left, st.right)]
#     st = st.left
#     while isinstance(st, Sequence):
#         suffix = Sequence(st.right, suffix)
#         st = st.left
#         res.append((st, suffix))

#     # convert splits into programs and normalize
#     res2 = [(normalize_program_structure(Program(p.inputs, x, p.output)), normalize_program_structure(Program(p.inputs, y, p.output))) for x, y in res]

#     # dead code elimination
#     res3 = []
#     for (l, r) in res2:
#         r_fv = get_free_variables(r)
#         st2 = typecheck.typecheck_statement(l.statement, p.inputs)
#         assert st2 is not None
#         r_inputs = {k: v for (k, v) in st2.items() if k in r_fv}
#         l_assigned = _updated_variables(l.statement)
#         l_res = Vector([VectorAccess(Variable(k), i) for k in r_inputs.keys() for i in range(r_inputs[k]) if k in l_assigned])
#         res3.append((Program(p.inputs, l.statement, l_res), Program(r_inputs, r.statement, p.output)))

#     return res3

def get_binary_splits(p: Program) -> List[Tuple[Program, Program]]:

    def get_binary_splits(s: Statement) -> List[Tuple[Statement, Statement]]:
        if not isinstance(s, Sequence):
            return [(Skip(), s), (s, Skip())]

        left_splits = get_binary_splits(s.left)
        left_splits = [(l, Sequence(r, s.right)) for (l, r) in left_splits]
        right_splits = get_binary_splits(s.right)
        right_splits = [(Sequence(s.left, l), r) for (l, r) in right_splits]

        splits = left_splits + right_splits
        splits = [(skip_remove(x), skip_remove(y)) for (x, y) in splits]

        final_splits = []
        seen = set()
        for x in splits:
            if str(x) in seen:
                continue
            seen.add(str(x))
            final_splits.append(x)

        return final_splits

    splits = get_binary_splits(p.statement)
    res = []
    for (l, r) in splits:
        l_def = typecheck.typecheck_statement(l, typecheck.TypeCheckResult(typecheck.inputs_to_types(p.inputs), {}))
        assert l_def is not None
        r_fv = get_free_variables(Program({}, r, p.outputs))

        r_inputs = collections.OrderedDict() # type: Dict[str, List[int]]
        for (i, n) in p.inputs.items():
            if i in r_fv:
                r_inputs[i] = n
        for (k, v) in l_def.vars.items():
            if k in p.inputs:
                continue
            if k in r_fv:
                r_inputs[k] = v.to_list()

        l_assigned = _updated_variables(l)

        l_res = [Variable(k) for k in r_inputs if k in l_assigned] # type: List[Expr]
        res.append((Program(p.inputs, l, l_res), Program(r_inputs, r, p.outputs)))

    return res


def get_free_variables(p: Program) -> Set[str]:
    def _free_variables_of_expr(e: Expr, defined_variables: Set[str]) -> Set[str]:
        if isinstance(e, Variable):
            return {e.name} - defined_variables
        elif isinstance(e, Value):
            return set()
        elif isinstance(e, Binop):
            return _free_variables_of_expr(e.left, defined_variables) | _free_variables_of_expr(e.right, defined_variables)
        elif isinstance(e, Unop):
            return _free_variables_of_expr(e.expr, defined_variables)
        elif isinstance(e, Vector):
            return set(x for y in e.exprs for x in _free_variables_of_expr(y, defined_variables))
        elif isinstance(e, VectorAccess):
            return _free_variables_of_expr(e.expr, defined_variables)
        else:
            raise ValueError(e)

    def _free_variables_of_statement(s: Statement, defined_variables: Set[str]) -> Tuple[Set[str], Set[str]]:
        # returns (free variables, defined variables)
        if isinstance(s, Sequence):
            l, l_defined = _free_variables_of_statement(s.left, defined_variables)
            r, r_defined = _free_variables_of_statement(s.right, l_defined)
            return l | r, r_defined
        elif isinstance(s, IfThen):
            raise NotImplementedError()
        elif isinstance(s, Assignment):
            fvs = _free_variables_of_expr(s.value, defined_variables)
            for i in s.indices:
                if isinstance(i, str):
                    fvs = fvs | {i}
            return fvs, defined_variables | {s.name}
        elif isinstance(s, Skip):
            return set(), defined_variables
        elif isinstance(s, Print):
            return _free_variables_of_expr(s.value, defined_variables), defined_variables
        elif isinstance(s, Iterate):
            it, it_defined = _free_variables_of_statement(s.s, defined_variables | {s.name})
            return it, defined_variables | it_defined
        else:
            raise ValueError(s)

    free, defined = _free_variables_of_statement(p.statement, set())
    return free
