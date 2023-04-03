from dataclasses import dataclass
from typing import Any, Set, Optional, Union, Mapping, List, Tuple
from enum import Enum

class BinaryOperator(Enum):
    ADD = 0
    MUL = 1
    MAX = 2
    MIN = 3
    DOT = 4
    POW = 5

class UnaryOperator(Enum):
    NEG = 0
    SIN = 1
    COS = 2
    ARCSIN = 3
    ARCCOS = 4
    LOG = 5
    EXP = 6
    SQRT = 7
    RECIP = 8
    RECIP1M = 9

@dataclass
class Variable:
    name: str

@dataclass
class Value:
    value: float

@dataclass
class Binop:
    op: BinaryOperator
    left: 'Expr'
    right: 'Expr'

@dataclass
class Pow:
    left: 'Expr'
    left_constraint: float
    right_constraint: float
    right: 'Expr'

@dataclass
class Unop:
    op: UnaryOperator
    expr: 'Expr'

@dataclass
class ConstrainedUnop:
    op: UnaryOperator
    constraint: float
    around: float
    expr: 'Expr'

@dataclass
class Vector:
    exprs: List['Expr']

@dataclass
class VectorAccess:
    expr: 'Expr'
    index: Union[str, int]

Expr = Union[Variable, Value, Binop, Pow, Unop, Vector, VectorAccess, ConstrainedUnop]

@dataclass
class Sequence:
    left: 'Statement'
    right: 'Statement'

@dataclass
class IfThen:
    condition: Expr
    left: 'Statement'
    right: 'Statement'

@dataclass
class Iterate:
    name: str
    start: Union[str, int]
    end: Union[str, int]
    s: 'Statement'

@dataclass
class Assignment:
    name: str
    indices: List[Union[str, int]]
    value: Expr

@dataclass
class Print:
    value: Expr

@dataclass
class Skip:
    pass

Statement = Union[Sequence, IfThen, Assignment, Skip, Print, Iterate]

@dataclass
class Program:
    inputs: Mapping[str, List[int]]
    statement: Statement
    outputs: List[Expr]
