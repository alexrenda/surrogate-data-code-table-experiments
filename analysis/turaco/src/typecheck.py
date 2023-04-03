from abc import ABC, abstractmethod
import copy
from typing import Any, Set, Optional, Union, Mapping, List, Tuple
from .syntax import *
import sys
from dataclasses import dataclass

class Type(ABC):
    @abstractmethod
    def to_list(self) -> List[int]:
        pass

@dataclass
class ScalarType(Type):
    def to_list(self) -> List[int]:
        return []

@dataclass
class VectorType(Type):
    inner: Type
    length: int

    def to_list(self) -> List[int]:
        inner_list = self.inner.to_list()
        inner_list.insert(0, self.length)
        return inner_list

@dataclass
class IterationVariableType(Type):
    min: int
    max: int

    def to_list(self) -> List[int]:
        raise Exception("IterationVariableType cannot be converted to a list")

@dataclass
class TypeCheckResult:
    vars: Mapping[str, Type]
    iter_vars: Mapping[str, IterationVariableType]

    def merge(self, other: 'TypeCheckResult') -> 'TypeCheckResult':
        vars = {**self.vars, **other.vars}
        # iter vars should never escape
        assert self.iter_vars == other.iter_vars
        return TypeCheckResult(vars, self.iter_vars)

def typecheck_expression(expression: Expr, ctx: TypeCheckResult) -> Optional[Type]:
    if isinstance(expression, Variable):
        if expression.name not in ctx.vars:
            print('Expression not in context: "{}"'.format(expression.name), file=sys.stderr)
            return None
        return ctx.vars[expression.name]
    elif isinstance(expression, Value):
        return ScalarType()
    elif isinstance(expression, Binop) or isinstance(expression, Pow):
        l = typecheck_expression(expression.left, ctx)
        r = typecheck_expression(expression.right, ctx)
        if l is None or r is None:
            return None
        if isinstance(expression, Pow) or expression.op in (BinaryOperator.ADD, BinaryOperator.MUL, BinaryOperator.MAX, BinaryOperator.MIN, BinaryOperator.POW):
            if isinstance(l, ScalarType):
                return r
            elif isinstance(r, ScalarType):
                return l
            elif l != r:
                return None
            else:
                return l
        elif expression.op == BinaryOperator.DOT:
            raise NotImplementedError()
        else:
            raise ValueError(expression.op)
    elif isinstance(expression, Unop) or isinstance(expression, ConstrainedUnop):
        return typecheck_expression(expression.expr, ctx)
    elif isinstance(expression, Vector):
        zz = None
        for ve in expression.exprs:
            tc = typecheck_expression(ve, ctx)
            if tc is None:
                return None

            if zz is None:
                zz = tc
            elif tc != zz:
                return None
        if zz is None:
            return None
        return VectorType(zz, len(expression.exprs))
    elif isinstance(expression, VectorAccess):
        ln = typecheck_expression(expression.expr, ctx)
        if ln is None:
            return None

        if not isinstance(ln, VectorType):
            return None

        if isinstance(expression.index, int):
            if expression.index < 0 or expression.index >= ln.length:
                print('Index out of bounds: "{}"'.format(expression), file=sys.stderr)
                return None
        elif isinstance(expression.index, str):
            if expression.index not in ctx.iter_vars:
                print('Expression not in context: "{}"'.format(expression.index), file=sys.stderr)
                return None

            if ctx.iter_vars[expression.index].min < 0 or ctx.iter_vars[expression.index].max > ln.length:
                print('Index out of bounds: "{}"'.format(expression), file=sys.stderr)
                return None
        else:
            raise ValueError(expression.index)

        return ln.inner
    else:
        raise NotImplementedError(expression)

def typecheck_statement(statement: Statement, ctx: TypeCheckResult) -> Optional[TypeCheckResult]:
    if isinstance(statement, Sequence):
        lctx = typecheck_statement(statement.left, ctx)
        if lctx is None:
            return None
        return typecheck_statement(statement.right, lctx)
    elif isinstance(statement, IfThen):
        cond = typecheck_expression(statement.condition, ctx)
        if cond is None:
            return None
        elif not isinstance(cond, ScalarType):
            print('Condition not a scalar: "{}"'.format(statement.condition), file=sys.stderr)
            return None

        lctx = typecheck_statement(statement.left, ctx)
        if lctx is None:
            return None
        rctx = typecheck_statement(statement.right, ctx)
        if rctx is None:
            return None
        return lctx.merge(rctx)
    elif isinstance(statement, Iterate):
        if isinstance(statement.start, str):
            if statement.start not in ctx.iter_vars:
                print('Expression not in context: "{}"'.format(statement.start), file=sys.stderr)
                return None
            itvart = ctx.vars[statement.start]
            if not isinstance(itvart, IterationVariableType):
                print('Expression not an iteration variable: "{}"'.format(statement.start), file=sys.stderr)
                return None
            start = itvart.min
        elif isinstance(statement.start, int):
            start = statement.start

        if isinstance(statement.end, str):
            if statement.end not in ctx.iter_vars:
                print('Expression not in context: "{}"'.format(statement.end), file=sys.stderr)
                return None
            itvart = ctx.vars[statement.end]
            if not isinstance(itvart, IterationVariableType):
                print('Expression not an iteration variable: "{}"'.format(statement.end), file=sys.stderr)
                return None
            end = itvart.max
        elif isinstance(statement.end, int):
            end = statement.end

        if start > end:
            print('Iteration start > end: "{}"'.format(statement), file=sys.stderr)
            return None

        return typecheck_statement(statement.s, TypeCheckResult(ctx.vars, {**ctx.iter_vars, statement.name: IterationVariableType(start, end)}))
    elif isinstance(statement, Assignment):
        e = typecheck_expression(statement.value, ctx)
        if e is None:
            return None

        if statement.indices:
            if statement.name not in ctx.vars:
                print('Variable not in context: "{}"'.format(statement.name), file=sys.stderr)
                return None

            expected_type = ctx.vars[statement.name]
            for i in statement.indices:
                assert isinstance(expected_type, VectorType)
                if isinstance(i, int):
                    if i < 0 or i >= expected_type.length:
                        print('Index out of bounds: "{}"'.format(statement), file=sys.stderr)
                        return None
                elif isinstance(i, str):
                    if i not in ctx.iter_vars:
                        print('Expression not in context: "{}"'.format(i), file=sys.stderr)
                        return None

                    if ctx.iter_vars[i].min < 0 or ctx.iter_vars[i].max > expected_type.length:
                        print('Index out of bounds: "{}"'.format(statement), file=sys.stderr)
                        return None

                expected_type = expected_type.inner

            if e != expected_type:
                print('Assignment type mismatch: "{}"'.format(statement), file=sys.stderr)
                return None
        else:
            ctx = TypeCheckResult({**ctx.vars, statement.name: e}, ctx.iter_vars)

        return ctx

    elif isinstance(statement, Skip):
        return ctx
    elif isinstance(statement, Print):
        print(f'{statement}: {ctx}')
        return ctx
    else:
        raise ValueError(statement)

def inputs_to_types(prog_inputs: Mapping[str, List[int]]) -> Mapping[str, Type]:
    inputs = {}
    for k in prog_inputs:
        last_type = ScalarType() # type: Type
        for t in prog_inputs[k][::-1]:
            last_type = VectorType(last_type, t)
        inputs[k] = last_type
    return inputs

def get_program_types(program: Program) -> Optional[TypeCheckResult]:
    inputs = inputs_to_types(program.inputs)
    ctx = TypeCheckResult(inputs, {})
    octx = typecheck_statement(program.statement, ctx)
    return octx

def typecheck_program(program: Program) -> Optional[List[Type]]:
    inputs = inputs_to_types(program.inputs)
    ctx = TypeCheckResult(inputs, {})
    octx = typecheck_statement(program.statement, ctx)
    if octx is None:
        return None

    return_types = []
    for o in program.outputs:
        typ = typecheck_expression(o, octx)
        if typ is None:
            return None
        return_types.append(typ)

    return return_types
