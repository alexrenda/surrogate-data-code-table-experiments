from typing import Any, Callable, Set, Optional, Union, Mapping, List, Tuple
from .syntax import *
from . import util
from . import typecheck
from . import serialize
import sys
import numpy as np

_param_counter = 0

def optimize_complexity_interpret_program(program: Program, max_scaling: Union[float, Mapping[str, List[float]]]) -> Tuple[float, float]:
    new_program_statement = program.statement
    for (k, n) in program.inputs.items():
        exprs = [] # type: List[Expr]
        for i in range(n):
            scale = max_scaling[k][i] if isinstance(max_scaling, dict) else max_scaling
            exprs.append(Binop(BinaryOperator.MUL, 0, VectorAccess(Variable(k), i), Value(scale)))
        new_program_statement = Sequence(Assignment(k, Vector(exprs)), new_program_statement)

    new_program_statement = Sequence(new_program_statement, Assignment('__output__', program.output))

    tcr = typecheck.typecheck_statement(new_program_statement, program.inputs)
    assert tcr is not None
    output_dim = tcr['__output__']
    output_sum_bop = VectorAccess(Variable('__output__'), 0) # type: Expr
    for i in range(1, output_dim):
        output_sum_bop = Binop(BinaryOperator.ADD, 0, output_sum_bop, VectorAccess(Variable('__output__'), i))
    new_program_statement = Sequence(new_program_statement, Assignment('__output_sum__', output_sum_bop))

    pytorch = translate_statement_to_pytorch(new_program_statement, program.inputs)

def translate_statement_to_pytorch_complexity_optimization(statement: Statement, var_type: Mapping[str, int], indent: int = 0) -> Tuple[str, List[str], Mapping[str, int]]:
    if isinstance(statement, Sequence):
        l_code, l_params, l_var_type = translate_statement_to_pytorch_complexity_optimization(statement.left, var_type, indent)
        r_code, r_params, r_var_type = translate_statement_to_pytorch_complexity_optimization(statement.right, l_var_type, indent)
        return l_code + '\n' + r_code, l_params + r_params, r_var_type
    elif isinstance(statement, IfThen):
        raise NotImplementedError()
    elif isinstance(statement, Assignment):
        a_name, a_value = statement.name, statement.value
        e, ep, params = translate_expr_to_pytorch_complexity_optimization(a_value, var_type)
        e_assignment = '{indent}{a_name} = {e}'.format(indent=' ' * indent, a_name=a_name, e=e)
        ep_assignment = '{indent}{a_name}__p = {ep}'.format(indent=' ' * indent, a_name=a_name, ep=ep)
        return e_assignment + '\n' + ep_assignment, params, var_type
    elif isinstance(statement, Skip):
        return '', [], var_type
    elif isinstance(statement, Print):
        p_value = statement.value
        e, ep, params = translate_expr_to_pytorch_complexity_optimization(p_value, var_type)
        return '{indent}print(e={e_ser}; e(.) = {e}; e\'(.) = {ep})'.format(indent=' ' * indent, e_ser=serialize.serialize_expr(p_value), e=e, ep=ep), params, var_type
    else:
        raise ValueError('Unknown statement type: {}'.format(statement))

# class BinaryOperator(Enum):
#     ADD = 0
#     MUL = 1
#     MAX = 2
#     MIN = 3
#     DOT = 4
#     POW = 5

# class UnaryOperator(Enum):
#     NEG = 0
#     SIN = 1
#     COS = 2
#     ARCSIN = 3
#     ARCCOS = 4
#     LOG = 5
#     EXP = 6
#     SQRT = 7
#     INV = 8


def translate_expr_to_pytorch_complexity_optimization(expr: Expr, var_type: Mapping[str, int], indent: int = 0) -> Tuple[str, str, List[str]]:
    global _param_counter

    if isinstance(expr, Variable):
        return expr.name, '{}__p'.format(expr.name), []
    elif isinstance(expr, Value):
        return str(abs(expr.value)), '0', []
    elif isinstance(expr, Binop):
        b_left = expr.left
        b_right = expr.right
        b_left_e, b_left_ep, b_left_params = translate_expr_to_pytorch_complexity_optimization(b_left, var_type)
        b_right_e, b_right_ep, b_right_params = translate_expr_to_pytorch_complexity_optimization(b_right, var_type)

        if expression.op == BinaryOperator.ADD:
            e = '({b_left_e} + {b_right_e})'.format(b_left_e=b_left_e, b_right_e=b_right_e)
            ep = '({b_left_ep} + {b_right_ep})'.format(b_left_ep=b_left_ep, b_right_ep=b_right_ep)
        elif expression.op == BinaryOperator.MUL:
            e = '({b_left_e} * {b_right_e})'.format(b_left_e=b_left_e, b_right_e=b_right_e)
            ep = '({b_left_e} * {b_right_ep} + {b_left_ep} * {b_right_e})'.format(b_left_e=b_left_e, b_right_e=b_right_e, b_left_ep=b_left_ep, b_right_ep=b_right_ep)
        elif expression.op == BinaryOperator.POW:


            new_expr = Unop(UnaryOperator.EXP, 0, 0, Binop(BinaryOperator.MUL, 0, b_right, Unop(UnaryOperator.LOG, 0, 0, Binop(BinaryOperator.ADD, 0, b_left, Value(-1)))))
            e, ep, params = translate_expr_to_pytorch_complexity_optimization(new_expr, var_type)

    elif isinstance(expr, Unop):
    elif isinstance(expr, Vector):
    elif isinstance(expr, VectorAccess):
    else:
        raise ValueError('Unknown expression type: {}'.format(expr))
