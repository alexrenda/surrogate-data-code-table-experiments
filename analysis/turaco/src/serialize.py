from .syntax import *

def serialize_expression(e: Expr) -> str:
    if isinstance(e, Variable):
        return e.name
    elif isinstance(e, Value):
        return str(e.value)
    elif isinstance(e, Binop):
        if e.op == BinaryOperator.ADD:
            ops = 'add'
        elif e.op == BinaryOperator.MUL:
            ops = 'mul'
        elif e.op == BinaryOperator.POW:
            ops = 'pow'
        else:
            raise ValueError(e.op)
        return '{}({}, {})'.format(ops, serialize_expression(e.left), serialize_expression(e.right))
    elif isinstance(e, Unop):
        if e.op == UnaryOperator.NEG:
            ops = 'neg'
        elif e.op == UnaryOperator.SIN:
            ops = 'sin'
        elif e.op == UnaryOperator.COS:
            ops = 'cos'
        elif e.op == UnaryOperator.ARCSIN:
            ops = 'arcsin'
        elif e.op == UnaryOperator.ARCCOS:
            ops = 'arccos'
        elif e.op == UnaryOperator.LOG:
            ops = 'log'
        elif e.op == UnaryOperator.EXP:
            ops = 'exp'
        elif e.op == UnaryOperator.SQRT:
            ops = 'sqrt'
        elif e.op == UnaryOperator.RECIP:
            ops = 'recip'
        else:
            raise ValueError(e.op)

        return '{}({})'.format(ops, serialize_expression(e.expr))
    elif isinstance(e, VectorAccess):
        return '{}[{}]'.format(serialize_expression(e.expr), e.index)
    elif isinstance(e, Vector):
        return '[{}]'.format(','.join(map(serialize_expression, e.exprs)))
    else:
        raise ValueError(e)


def serialize_statement(s: Statement, indent:int=0) -> str:
    if isinstance(s, Sequence):
        return serialize_statement(s.left, indent) + '\n' + serialize_statement(s.right, indent)
    elif isinstance(s, IfThen):
        l1 = ' '*indent + 'if ({} > 0) {{\n'.format(serialize_expression(s.condition))
        l2 = serialize_statement(s.left, indent=2+indent) + '\n'
        l3 = ' '*indent + '} {\n'
        l4 = serialize_statement(s.right, indent=2+indent) + '\n'
        l5 = ' '*indent + '}'
        return l1+l2+l3+l4+l5
    elif isinstance(s, Assignment):
        idxs = ''.join('[{}]'.format(i) for i in s.indices)
        return ' '*indent + '{}{} := {}'.format(s.name, idxs, serialize_expression(s.value))
    elif isinstance(s, Skip):
        return ' '*indent + 'skip'
    elif isinstance(s, Print):
        return ' '*indent + 'print({})'.format(serialize_expression(s.value))
    elif isinstance(s, Iterate):
        return ' '*indent + 'iterate {} from {} to {} {{\n'.format(s.name, s.start, s.end) + serialize_statement(s.s, indent = 2+indent) + '\n' + ' '*indent + '}'
    else:
        raise ValueError(s)

def serialize_program(program: Program) -> str:
    params = ', '.join('{}{}'.format(k, ''.join('[{}]'.format(x) for x in v)) for (k, v) in program.inputs.items())
    body = serialize_statement(program.statement, 2)
    return_statement = ', '.join(serialize_expression(e) for e in program.outputs)
    return 'fun({}) {{\n{}\n  return {}\n}}'.format(params, body, return_statement)
