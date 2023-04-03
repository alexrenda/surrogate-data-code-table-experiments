from typing import Any, Set, Optional, Union, Mapping, List, Tuple, Callable
from . import serialize
from .syntax import *
import numpy as np
import sys
import functools
from .util import Matrix, _apply_binary_operator, _apply_unary_operator, _assert_predicate, _reduce
from dataclasses import dataclass
import copy

@dataclass
class Interval:
    l: float
    r: float

    def __add__(self, other: 'Interval') -> 'Interval':
        return Interval(self.l + other.l, self.r + other.r)

    def __mul__(self, other: 'Interval') -> 'Interval':
        return Interval(
            min(self.l * other.l, self.l * other.r, self.r * other.l, self.r * other.r),
            max(self.l * other.l, self.l * other.r, self.r * other.l, self.r * other.r)
        )

    def __truediv__(self, other: 'Interval') -> 'Interval':
        return Interval(
            min(self.l / other.l, self.l / other.r, self.r / other.l, self.r / other.r),
            max(self.l / other.l, self.l / other.r, self.r / other.l, self.r / other.r)
        )

    def __pow__(self, other: 'Interval') -> 'Interval':
        return Interval(
            min(self.l ** other.l, self.l ** other.r, self.r ** other.l, self.r ** other.r),
            max(self.l ** other.l, self.l ** other.r, self.r ** other.l, self.r ** other.r)
        )

    def __neg__(self) -> 'Interval':
        return Interval(-self.r, -self.l)

    def __str__(self) -> str:
        return '[{}, {}]'.format(self.l, self.r)

    def __repr__(self) -> str:
        return str(self)

    def sin(self) -> 'Interval':
        return Interval(
            min(np.sin(self.l), np.sin(self.r)),
            max(np.sin(self.l), np.sin(self.r))
        )

    def cos(self) -> 'Interval':
        return Interval(
            min(np.cos(self.l), np.cos(self.r)),
            max(np.cos(self.l), np.cos(self.r))
        )

    def exp(self) -> 'Interval':
        return Interval(np.exp(self.l), np.exp(self.r))

    def log(self) -> 'Interval':
        return Interval(np.log(self.l), np.log(self.r))

@dataclass
class DualInterval:
    l: Interval
    r: Interval

    # addition
    def __add__(self, other: 'DualInterval') -> 'DualInterval':
        return DualInterval(self.l + other.l, self.r + other.r)

    # multiplication
    def __mul__(self, other: 'DualInterval') -> 'DualInterval':
        return DualInterval(
            self.l * other.l,
            self.l * other.r + self.r * other.l
        )

    # exponentiation
    def __pow__(self, other: 'DualInterval') -> 'DualInterval':
        return DualInterval(
            self.l ** other.l,
            self.l ** other.r * other.l * self.r
        )

    def sin(self) -> 'DualInterval':
        return DualInterval(
            self.l.sin(),
            self.r * self.l.cos()
        )

    def cos(self) -> 'DualInterval':
        return DualInterval(
            self.l.cos(),
            -self.r * self.l.sin()
        )

    def exp(self) -> 'DualInterval':
        return DualInterval(
            self.l.exp(),
            self.r * self.l.exp()
        )

    def log(self) -> 'DualInterval':
        return DualInterval(
            self.l.log(),
            self.r / self.l
        )

    def log1p(self) -> 'DualInterval':
        # log(1+x)
        return DualInterval(self.l + Interval(1, 1), self.r).log()

JacVal = Matrix[DualInterval]

def jacobian_interpret_expression(expression: Expr, state: Mapping[str, JacVal], ivar_state: Mapping[str, int]) -> JacVal:
    if isinstance(expression, Variable):
        return state[expression.name]
    elif isinstance(expression, Value):
        return DualInterval(Interval(expression.value, expression.value), Interval(0, 0))
    elif isinstance(expression, Binop):
        op = expression.op
        a = jacobian_interpret_expression(expression.left, state, ivar_state)
        b = jacobian_interpret_expression(expression.right, state, ivar_state)

        bop = lambda l, r: l # type: Callable[[DualInterval, DualInterval], DualInterval]

        if op == BinaryOperator.ADD:
            bop = lambda l, r: l + r
        elif op == BinaryOperator.MUL:
            bop = lambda l, r: l * r
        elif op == BinaryOperator.POW:
            bop = lambda l, r: l ** r
        else:
            raise NotImplementedError('Binary operator {} not implemented'.format(op))

        return _apply_binary_operator(a, b, bop)
    elif isinstance(expression, Unop):
        uop = expression.op
        a = jacobian_interpret_expression(expression.expr, state, ivar_state)

        uuop = lambda l: l # type: Callable[[DualInterval], DualInterval]

        if uop == UnaryOperator.NEG:
            return jacobian_interpret_expression(Binop(BinaryOperator.MUL, Value(-1), expression.expr), state, ivar_state)
        elif uop == UnaryOperator.SIN:
            uuop = lambda l: l.sin()
        elif uop == UnaryOperator.COS:
            uuop = lambda l: l.cos()
        elif uop == UnaryOperator.EXP:
            uuop = lambda l: l.exp()
        elif uop == UnaryOperator.LOG:
            uuop = lambda l: l.log1p()
        else:
            raise NotImplementedError()

        return _apply_unary_operator(a, uuop)

    elif isinstance(expression, Vector):
        return [jacobian_interpret_expression(e, state, ivar_state) for e in expression.exprs]
    elif isinstance(expression, VectorAccess):
        exp = jacobian_interpret_expression(expression.expr, state, ivar_state)
        assert isinstance(exp, list), '{} --> {}'.format(expression.expr, exp)
        if isinstance(expression.index, str):
            return exp[ivar_state[expression.index]]
        else:
            return exp[expression.index]
    else:
        raise NotImplementedError()

def jacobian_interpret_statement(statement: Statement, state: Mapping[str, JacVal], ivar_state: Mapping[str, int]) -> Mapping[str, JacVal]:
    if isinstance(statement, Sequence):
        state = jacobian_interpret_statement(statement.left, state, ivar_state)
        state = jacobian_interpret_statement(statement.right, state, ivar_state)
        return state
    elif isinstance(statement, IfThen):
        raise NotImplementedError()
    elif isinstance(statement, Assignment):
        if statement.indices:
            state = copy.deepcopy(state)
            var = state[statement.name] # type: Matrix[DualInterval]
            for i in statement.indices[:-1]:
                if isinstance(i, str):
                    var = var[ivar_state[i]] # type: ignore
                else:
                    var = var[i] # type: ignore
            res = jacobian_interpret_expression(statement.value, state, ivar_state)
            final_index = statement.indices[-1]
            if isinstance(final_index, str):
                var[ivar_state[final_index]] = res # type: ignore
            else:
                var[final_index] = res # type: ignore
            return state
        else:
            return {**state, statement.name: jacobian_interpret_expression(statement.value, state, ivar_state)}

        return {**state, statement.name: jacobian_interpret_expression(statement.value, state, ivar_state)}
    elif isinstance(statement, Skip):
        return state
    elif isinstance(statement, Print):
        print('{}: {}'.format(serialize.serialize_expression(statement.value), jacobian_interpret_expression(statement.value, state, ivar_state)))
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
            state = jacobian_interpret_statement(statement.s, state, ivar_state)
        return state

    else:
        raise NotImplementedError(statement)

ProgramJacobian = List[List[Interval]]

def dual_interval_to_interval(x: Matrix[DualInterval], side: str='l') -> Matrix[Interval]:
    if isinstance(x, DualInterval):
        if side == 'l':
            return x.l
        elif side == 'r':
            return x.r
        else:
            raise ValueError('side must be l or r')
    elif isinstance(x, list):
        return [dual_interval_to_interval(e, side) for e in x]

def interval_to_float(x: Matrix[Interval], side: str='l') -> Matrix[float]:
    if isinstance(x, Interval):
        if side == 'l':
            return x.l
        elif side == 'r':
            return x.r
        else:
            raise ValueError('side must be l or r')
    elif isinstance(x, list):
        return [interval_to_float(e, side) for e in x]

def _flatten(x: Matrix[Interval]) -> List[Interval]:
    if isinstance(x, list):
        return sum([_flatten(e) for e in x], [])
    else:
        return [x]

def jacobian_interpret_program(program: Program, inputs: Mapping[str, Matrix[Interval]]) -> Tuple[Mapping[str, JacVal], ProgramJacobian]:
    # forward mode AD in the dual interval domain
    def build_input_matrix(x: Matrix[Interval]) -> Matrix[DualInterval]:
        if isinstance(x, Interval):
            return DualInterval(x, Interval(0., 0.))
        elif isinstance(x, list):
            return [build_input_matrix(e) for e in x]

    def get_jacval_mtx(x: Matrix[DualInterval], input_mtx_setter: Callable[[DualInterval], None]) -> List[List[Interval]]:
        if isinstance(x, DualInterval):
            input_mtx_setter(DualInterval(x.l, Interval(1., 1.)))
            state = jacobian_interpret_statement(program.statement, dual_inputs, {})
            outputs = _flatten([dual_interval_to_interval(jacobian_interpret_expression(e, state, {}), 'r') for e in program.outputs])

            input_mtx_setter(DualInterval(x.l, Interval(0., 0.)))
            return [outputs]
        elif isinstance(x, list):
            res = []
            for i, e in enumerate(x):
                def set_input_mtx(v: DualInterval) -> None:
                    x[i] = v # type: ignore
                res.extend(get_jacval_mtx(e, set_input_mtx))
            return res
        else:
            raise NotImplementedError(x)

    dual_inputs = {i: build_input_matrix(x) for i, x in inputs.items()}
    state = jacobian_interpret_statement(program.statement, dual_inputs, {})

    res = []
    for (k, v) in dual_inputs.items():
        def m_setter(v: DualInterval) -> None:
            dual_inputs[k] = v
        res.extend(get_jacval_mtx(v, m_setter))

    return state, res
