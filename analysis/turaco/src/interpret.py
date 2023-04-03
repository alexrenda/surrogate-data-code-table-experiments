from typing import Any, Callable, Set, Optional, Union, Mapping, List, Tuple
import numpy as np
import sys
import copy

from .syntax import *
from .util import Matrix, _apply_binary_operator, _apply_unary_operator, _assert_predicate
from . import serialize

IValue = Matrix[float]

def interpret_expression(expression: Expr, state: Mapping[str, IValue], ivar_state: Mapping[str, int]) -> IValue:
    if isinstance(expression, Variable):
        return state[expression.name]
    elif isinstance(expression, Value):
        return [expression.value]
    elif isinstance(expression, Binop) or isinstance(expression, Pow):
        left = expression.left
        right = expression.right

        if isinstance(expression, Binop):
            mop = expression.op
        elif isinstance(expression, Pow):
            mop = BinaryOperator.POW
        else:
            raise ValueError()

        left_val = interpret_expression(left, state, ivar_state)
        right_val = interpret_expression(right, state, ivar_state)
        if mop in (BinaryOperator.ADD, BinaryOperator.MUL, BinaryOperator.MAX, BinaryOperator.MIN, BinaryOperator.POW):
            boperator = None # type: Optional[Callable[[float, float], float]]
            if mop == BinaryOperator.ADD:
                boperator = lambda l, r: l + r
            elif mop == BinaryOperator.MUL:
                boperator = lambda l, r: l * r
            elif mop == BinaryOperator.MAX:
                boperator = max
            elif mop == BinaryOperator.MIN:
                boperator = min
            elif mop == BinaryOperator.POW:
                boperator = lambda l, r: l**r # type: ignore
            else:
                raise ValueError(mop)

            return _apply_binary_operator(left_val, right_val, boperator)
        else:
            raise ValueError(expression.op)

    elif isinstance(expression, Unop) or isinstance(expression, ConstrainedUnop):
        op = expression.op
        val = interpret_expression(expression.expr, state, ivar_state)
        uoperator = None # type: Optional[Callable[[float], float]]
        if op == UnaryOperator.NEG:
            uoperator = lambda x: -x
        elif op == UnaryOperator.SIN:
            uoperator = np.sin
        elif op == UnaryOperator.COS:
            uoperator = np.cos
        elif op == UnaryOperator.ARCSIN:
            uoperator = np.arcsin
        elif op == UnaryOperator.ARCCOS:
            uoperator = np.arccos
        elif op == UnaryOperator.LOG:
            # _assert_predicate(val, lambda x: -1 < x <= 1)
            # def m_log(v: float) -> float:
            #     assert -1<v<=1
                # return float(np.log(v+1))
            uoperator = np.log
        elif op == UnaryOperator.EXP:
            uoperator = np.exp
        elif op == UnaryOperator.SQRT:
            uoperator = np.sqrt
        elif op == UnaryOperator.RECIP:
            _assert_predicate(val, lambda x: x > 0)
            uoperator = lambda x: 1/x
        else:
            raise ValueError(op)
        return _apply_unary_operator(val, uoperator)
    elif isinstance(expression, Vector):
        return [interpret_expression(e, state, ivar_state) for e in expression.exprs]
    elif isinstance(expression, VectorAccess):
        vexp = interpret_expression(expression.expr, state, ivar_state)
        assert isinstance(vexp, list), (expression, vexp)
        if isinstance(expression.index, str):
            return vexp[ivar_state[expression.index]]
        else:
            return vexp[expression.index]
    else:
        raise ValueError(expression)

def interpret_statement(statement: Statement, state: Mapping[str, IValue], ivar_state: Mapping[str, int], path: Optional[List[str]]=None) -> Mapping[str, IValue]:
    if isinstance(statement, Sequence):
        left_state = interpret_statement(statement.left, state, ivar_state, path=path)
        if isinstance(left_state, dict):
            return interpret_statement(statement.right, left_state, ivar_state, path=path)
        else:
            return left_state
    elif isinstance(statement, IfThen):
        condition = statement.condition
        left = statement.left
        right = statement.right

        val = interpret_expression(condition, state, ivar_state)

        if isinstance(val, list):
            assert len(val) == 1 and isinstance(val[0], float)
            val = val[0]
        else:
            assert isinstance(val, float)

        if val > 0:
            if path is not None:
                path.append('l')
            return interpret_statement(left, state, ivar_state, path=path)
        elif val <= 0:
            if path is not None:
                path.append('r')
            return interpret_statement(right, state, ivar_state, path=path)
        else:
            raise ValueError()
            return None
    elif isinstance(statement, Assignment):
        if statement.indices:
            state = copy.deepcopy(state)
            var = state[statement.name] # type: Matrix[Complexity]
            for i in statement.indices[:-1]:
                if isinstance(i, str):
                    var = var[ivar_state[i]] # type: ignore
                else:
                    var = var[i] # type: ignore
            res = interpret_expression(statement.value, state, ivar_state)
            final_index = statement.indices[-1]
            if isinstance(final_index, str):
                var[ivar_state[final_index]] = res # type: ignore
            else:
                var[final_index] = res # type: ignore
            return state
        else:
            return {**state, statement.name: interpret_expression(statement.value, state, ivar_state)}
    elif isinstance(statement, Skip):
        return state
    elif isinstance(statement, Print):
        print('{}: {}'.format(serialize.serialize_expression(statement.value), interpret_expression(statement.value, state, ivar_state)), file=sys.stderr)
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
            state = interpret_statement(statement.s, state, ivar_state, path=path)
        return state

    else:
        raise ValueError(statement)

def interpret_program(program: Program, inputs: Mapping[str, IValue], path: Optional[List[str]]=None) -> List[IValue]:
    state = interpret_statement(program.statement, inputs, {}, path=path)
    res = [interpret_expression(o, state, {}) for o in program.outputs]
    return res
