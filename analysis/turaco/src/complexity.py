from abc import ABC
import copy
from typing import Any, Callable, Set, Optional, Union, Mapping, List, Tuple
from .syntax import *
from . import util
from . import interpret
from . import typecheck
from . import serialize
from .util import Matrix, _apply_binary_operator, _apply_unary_operator, _assert_predicate
import sys
import numpy as np
import functools
import numbers

ComplexityValue = Tuple[float, float]
Complexity = Matrix[ComplexityValue]

def complexity_interpret_expression(expression: Expr, state: Mapping[str, Complexity], ivar_state: Mapping[str, int]) -> Complexity:
    pred = lambda c: True # type: Callable[[ComplexityValue], bool]
    uop = lambda v: (abs(v[0]), abs(v[1])) # type: Callable[[ComplexityValue], ComplexityValue]

    if isinstance(expression, Variable):
        return state[expression.name]
    elif isinstance(expression, Value):
        return (abs(expression.value), 0)
    elif isinstance(expression, Binop):
        left = expression.left
        right = expression.right
        left_val = complexity_interpret_expression(left, state, ivar_state)
        right_val = complexity_interpret_expression(right, state, ivar_state)

        if expression.op in (BinaryOperator.ADD, BinaryOperator.MUL, BinaryOperator.MAX, BinaryOperator.MIN, BinaryOperator.POW):
            op = lambda l, r: (l[0] + r[0], l[1] + r[1]) # type: Callable[[ComplexityValue, ComplexityValue], ComplexityValue]
            if expression.op == BinaryOperator.ADD:
                def op(l: ComplexityValue, r: ComplexityValue) -> ComplexityValue: return (l[0] + r[0], l[1] + r[1])
            elif expression.op == BinaryOperator.MUL:
                def op(l: ComplexityValue, r: ComplexityValue) -> ComplexityValue: return (l[0] * r[0], l[0] * r[1] + l[1] * r[0])
            # elif expression.op == BinaryOperator.POW:
            #     # pow(x, y) = exp(y * log(sub(x, 1)))
            #     i1 = Binop(BinaryOperator.ADD, 0, left, Value(-1))
            #     i2 = Unop(UnaryOperator.LOG, 0, 0, i1)
            #     i3 = Binop(BinaryOperator.MUL, 0, right, i2)
            #     i4 = Unop(UnaryOperator.EXP, 0, 0, i3)
            #     return complexity_interpret_expression(i4, state)
            #     raise NotImplementedError(expression.op)
            else:
                raise ValueError(expression.op)

            return _apply_binary_operator(left_val, right_val, op)
        else:
            raise ValueError(expression.op)
    elif isinstance(expression, ConstrainedUnop):
        expr = expression.expr
        val = complexity_interpret_expression(expr, state, ivar_state)

        if expression.op == UnaryOperator.LOG:
            c = expression.constraint
            def pred(v: ComplexityValue) -> bool: return v[1] == 0 or (v[0] >= c and 2*c > v[0])
            def uop(v: ComplexityValue) -> ComplexityValue:
                if v[1] == 0:
                    return (np.log(c), 0)
                zz = v[0] - c
                num = np.abs(np.log(c)) + np.log(c) - np.log(c - zz)
                denom = v[1] / (c - zz)
                return (num, denom)
        elif expression.op == UnaryOperator.RECIP:
            c = expression.constraint
            def pred(v: ComplexityValue) -> bool:
                if not (v[1] == 0 or (v[0] >= c and 2*c > v[0])):
                    print(v)
                return v[1] == 0 or (v[0] >= c and 2*c > v[0])
            def uop(v: ComplexityValue) -> ComplexityValue:
                if v[1] == 0:
                    return (1/c, 0)

                zz = v[0] - c
                num = 1/(c - zz)
                denom = v[1] / (c - zz)**2
                return (num, denom)
        elif expression.op == UnaryOperator.SQRT:
            c = expression.constraint
            a = expression.around
            assert a >= c
            def pred(v: ComplexityValue) -> bool:
                return True
                return v[1] == 0 or (v[0] >= c and 2*c > v[0])
            def uop(v: ComplexityValue) -> ComplexityValue:
                if v[1] == 0:
                    return (np.sqrt(c), 0)

                zz = v[0] - c
                # around `a - c`
                zz = zz + (a - c)

                assert zz >= 0 and zz < a

                num = np.sqrt(a - zz)
                denom = v[1] / (2 * (a - zz))

                return (num, denom)
        else:
            raise ValueError(expression.op)

        _assert_predicate(val, pred)
        return _apply_unary_operator(val, uop)
    elif isinstance(expression, Unop):
        expr = expression.expr
        val = complexity_interpret_expression(expr, state, ivar_state)


        if expression.op == UnaryOperator.NEG:
            def uop(v: ComplexityValue) -> ComplexityValue: return (v[0], v[1])
        elif expression.op == UnaryOperator.SIN:
            def uop(v: ComplexityValue) -> ComplexityValue: return (np.sinh(v[0]), v[1] * np.cosh(v[0]))
        elif expression.op == UnaryOperator.COS:
            def uop(v: ComplexityValue) -> ComplexityValue: return (np.cosh(v[0]), v[1] * np.sinh(v[0]))
        elif expression.op == UnaryOperator.ARCSIN:
            def uop(v: ComplexityValue) -> ComplexityValue: return (np.arcsin(v[0]), v[1] / np.sqrt(1 - v[0]**2))
        elif expression.op == UnaryOperator.ARCCOS:
            def uop(v: ComplexityValue) -> ComplexityValue: return (np.arcsin(v[0]) + np.pi/2, v[1] / np.sqrt(1 - v[0]**2))
        # elif expression.op == UnaryOperator.LOG:
        #     def pred(v: ComplexityValue) -> bool: return -1 <= v[0] < 1
            # def uop(v: ComplexityValue) -> ComplexityValue: return (-np.log(1 - v[0]), v[1] / (1 - v[0]))
        # elif expression.op == UnaryOperator.LOG1B:
        #     def uop(v: ComplexityValue) -> ComplexityValue:
        #         if v[0] < 2:
        #             c = 1.0
        #         else:
        #             c = 2*v[0]
        #         return (np.abs(np.log(c)) + np.log(c) + np.log(c - v[0]), v[1] / (c - v[0]))
        elif expression.op == UnaryOperator.LOG:
            assert False
            def pred(v: ComplexityValue) -> bool: return v[1] == 0
            def uop(v: ComplexityValue) -> ComplexityValue:
                num = np.abs(np.log(v[0]))
                denom = 0
                return (num, denom)
        elif expression.op == UnaryOperator.RECIP:
            def pred(v: ComplexityValue) -> bool: return v[1] == 0
            def uop(v: ComplexityValue) -> ComplexityValue:
                num = 1/v[0]
                denom = 0
                return (num, denom)

        elif expression.op == UnaryOperator.EXP:
            def uop(v: ComplexityValue) -> ComplexityValue: return (np.exp(v[0]), v[1] * np.exp(v[0]))
        elif expression.op == UnaryOperator.SQRT:
            raise NotImplementedError(op)
            # return [(a ** np.log(1/epsilon), np.log(1/epsilon)*b*a**(np.log(1/epsilon)-1)) for (a, b) in val]
        elif expression.op == UnaryOperator.RECIP:
            raise NotImplementedError(op)
            # return [(1 / a, -b / a**2) for (a, b) in val]
        elif expression.op == UnaryOperator.RECIP1M:
            raise NotImplementedError(op)
            # return [(1 / (1 - a), b / (1 - a)**2) for (a, b) in val]
            # raise NotImplementedError()
            # d = (np.log(epsilon) + np.log(1 - gamma)) / np.log(gamma)
            # ares = (a**(d+1) - 1) / (a - 1)
            # bres = (d * a**(d+1) - (d+1)*a**d + 1) / (a-1)**2
            # return (ares, bres * b)
        else:
            raise ValueError(expression.op)

        _assert_predicate(val, pred)
        res = _apply_unary_operator(val, uop)
        return res

    elif isinstance(expression, Pow):
        left, lconstraint, rconstraint, right = expression.left, expression.left_constraint, expression.right_constraint, expression.right
        lval, lderiv = complexity_interpret_expression(left, state, ivar_state)
        rval, rderiv = complexity_interpret_expression(right, state, ivar_state)
        assert lval >= lconstraint
        assert rval >= rconstraint

        import scipy.special
        cval = np.exp(lval - 1) * scipy.special.gamma(rval + 1)
        cderiv = cval * (lderiv + rderiv * scipy.special.psi(rval + 1))


        # cval = np.exp((1 + lval) * rval)
        # cderiv = np.exp((1 + lval) * rval) * (rval + (1 + lval) * rderiv)
        return cval, cderiv

    elif isinstance(expression, Vector):
        return [complexity_interpret_expression(e, state, ivar_state) for e in expression.exprs]
    elif isinstance(expression, VectorAccess):
        vec = complexity_interpret_expression(expression.expr, state, ivar_state)
        assert isinstance(vec, list), expression.expr
        if isinstance(expression.index, str):
            return vec[ivar_state[expression.index]]
        else:
            return vec[expression.index]
    else:
        raise NotImplementedError()

def complexity_interpret_statement(statement: Statement, state: Mapping[str, Complexity], ivar_state: Mapping[str, int]) -> Mapping[str, Complexity]:

    if isinstance(statement, Sequence):
        left_state = complexity_interpret_statement(statement.left, state, ivar_state)
        return complexity_interpret_statement(statement.right, left_state, ivar_state)
    elif isinstance(statement, Assignment):
        if statement.indices:
            state = copy.deepcopy(state)
            var = state[statement.name] # type: Matrix[Complexity]
            for i in statement.indices[:-1]:
                if isinstance(i, str):
                    var = var[ivar_state[i]] # type: ignore
                else:
                    var = var[i] # type: ignore
            res = complexity_interpret_expression(statement.value, state, ivar_state)
            final_index = statement.indices[-1]
            if isinstance(final_index, str):
                var[ivar_state[final_index]] = res # type: ignore
            else:
                var[final_index] = res # type: ignore
            return state
        else:
            return {**state, statement.name: complexity_interpret_expression(statement.value, state, ivar_state)}
    elif isinstance(statement, Skip):
        return state
    elif isinstance(statement, Print):
        print('{}: {}'.format(serialize.serialize_expression(statement.value), complexity_interpret_expression(statement.value, state, ivar_state)), file=sys.stdout)
        return state
    elif isinstance(statement, Iterate):
        if isinstance(statement.start, str):
            start = ivar_state[statement.start]
        else:
            start = statement.start

        if isinstance(statement.end, str):
            end = ivar_state[statement.end]
        else:
            end = statement.end

        for i in range(start, end):
            ivar_state = {**ivar_state, statement.name: i}
            state = complexity_interpret_statement(statement.s, state, ivar_state)
        return state

    else:
        raise ValueError(statement)

def complexity_interpret_program(program: Program, max_scaling: Mapping[str, interpret.IValue]) -> List[Complexity]:
    # for each program input, scale the input to its maximum value

    def scale_input(name: str, indices: List[Union[str, int]], scaling: interpret.IValue) -> Statement:
        if isinstance(scaling, float):
            v = Variable(name) # type: Expr
            for i in indices:
                v = VectorAccess(v, i)
            return Assignment(name, indices, Binop(BinaryOperator.MUL, v, Value(scaling)))
        else:
            assgns = [scale_input(name, indices + [i], scaling[i]) for i in range(len(scaling))]
            st = Skip() # type: Statement
            for a in assgns:
                st = Sequence(st, a)
            return st

    scales = [scale_input(name, [], scaling) for name, scaling in max_scaling.items()]
    scaled_program = Skip() # type: Statement
    for s in scales:
        scaled_program = Sequence(scaled_program, s)
    new_program_statement = Sequence(scaled_program, program.statement)

    input_types = typecheck.inputs_to_types(program.inputs)

    def build_input_matrix(x: typecheck.Type) -> Complexity:
        if isinstance(x, typecheck.ScalarType):
            return (1, 1)
        else:
            assert isinstance(x, typecheck.VectorType)
            length = x.length
            return [build_input_matrix(x.inner) for _ in range(length)]

    inputs = {name: build_input_matrix(t) for name, t in input_types.items()}
    cmap = complexity_interpret_statement(new_program_statement, inputs, {})
    output_exprs = [complexity_interpret_expression(e, cmap, {}) for e in program.outputs]

    return output_exprs

def sum_complexities(c: List[Complexity]) -> Tuple[float, float]:
    def sum_complexity(c: Complexity) -> Tuple[float, float]:
        if isinstance(c, tuple):
            return c
        else:
            cs = [sum_complexity(e) for e in c]
            return sum([v for v, _ in cs]), sum([g for _, g in cs])

    complexity = [sum_complexity(e) for e in c]
    return sum([v for v, _ in complexity]), sum([g for _, g in complexity])

def flat_scaling(p: Program, v: float) -> Mapping[str, interpret.IValue]:
    def build_scaling(t: typecheck.Type) -> interpret.IValue:
        if isinstance(t, typecheck.ScalarType):
            return v
        else:
            assert isinstance(t, typecheck.VectorType)
            return [build_scaling(t.inner) for _ in range(t.length)]

    return {name: build_scaling(t) for name, t in typecheck.inputs_to_types(p.inputs).items()}
