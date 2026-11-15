from __future__ import annotations

from typing import Any

import load_previous
import ast_nodes
from catalogue import (
    RULES,
    ensure_semantics,
    get_node_symbol,
    type_name_of,
)
from ir import (
    Add,
    Addi,
    Beq,
    Bge,
    Blt,
    Bne,
    Bnez,
    Call,
    Context,
    Div,
    FunctionIR,
    Jump,
    Label,
    LabelInstr,
    Li,
    Move,
    Mul,
    Neg,
    Param,
    ProgramIR,
    Rem,
    Return,
    Seqz,
    Slli,
    Slt,
    Snez,
    Sub,
    VReg,
    Xor,
)
import patterns
from patterns import COND, EFFECT, REG
from selector import Choice, SelectionTable, Selector


def _span_of(node: Any) -> Any:
    return getattr(node, "span", getattr(node, "origin", None))


def _extract_signed_int(node: Any) -> int:
    for helper_name in ("literal_value", "signed_value", "extract_int", "int_value"):
        if hasattr(patterns, helper_name):
            return int(getattr(patterns, helper_name)(node))
    if hasattr(node, "value") and isinstance(node.value, int):
        return int(node.value)
    operand = getattr(node, "operand", getattr(node, "expr", None))
    if operand is not None and hasattr(operand, "value"):
        return -int(operand.value)
    ch = patterns.children(node)
    if ch and hasattr(ch[0], "value"):
        return -int(ch[0].value)
    raise ValueError(f"Não foi possível extrair inteiro do literal em {_span_of(node)}")


def _extract_pow2_shift(node: Any) -> int:
    for helper_name in ("pow2_shift", "exponent_of_pow2", "log2_exact", "pow2_exponent"):
        if hasattr(patterns, helper_name):
            return int(getattr(patterns, helper_name)(node))
    val = _extract_signed_int(node)
    return val.bit_length() - 1


def _decode_escapes(text: str) -> str:
    out: list[str] = []
    i = 0
    escapes = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "0": "\0"}
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text) and text[i + 1] in escapes:
            out.append(escapes[text[i + 1]])
            i += 2
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


class CodeGenerator:
    def __init__(self, rules: Any = RULES) -> None:
        if callable(rules):
            rules = rules()
        self.rules = tuple(rules)
        self.selector = Selector(self.rules)

    def generate(self, program: Any) -> ProgramIR:
        ensure_semantics(program)
        program_ir = ProgramIR(functions=[], strings={})
        functions = getattr(program, "functions", getattr(program, "declarations", []))
        for func_decl in functions:
            if type(func_decl).__name__ == "FunctionDecl":
                func_ir = self._generate_function(func_decl, program_ir)
                program_ir.functions.append(func_ir)
        return program_ir

    def _generate_function(self, func_decl: Any, program_ir: ProgramIR) -> FunctionIR:
        func_symbol = get_node_symbol(func_decl)
        func_ir = FunctionIR(function=func_symbol, instructions=[])
        ctx = Context(func_ir, program_ir)

        params = getattr(func_decl, "params", getattr(func_decl, "parameters", []))
        for idx, param in enumerate(params):
            param_sym = get_node_symbol(param)
            dst_vreg = ctx.variable(param_sym)
            ctx.emit(Param(dst=dst_vreg, index=idx, origin=_span_of(param)))

        body = getattr(func_decl, "body", None)
        if body is not None:
            self._generate_stmt(body, ctx)

        ret_type = type_name_of(func_decl)
        if ret_type == "void":
            if not func_ir.instructions or not isinstance(func_ir.instructions[-1], Return):
                ctx.emit(Return(value=None, origin=_span_of(func_decl)))

        return func_ir

    def _generate_stmt(self, stmt: Any, ctx: Context) -> None:
        if stmt is None:
            return

        kind = type(stmt).__name__

        if kind in ("Block", "BlockStmt"):
            stmts = getattr(stmt, "statements", getattr(stmt, "body", []))
            for s in stmts:
                self._generate_stmt(s, ctx)
            return

        if kind in ("VarDecl", "VarDeclStmt"):
            init_expr = getattr(stmt, "initializer", getattr(stmt, "init", getattr(stmt, "value", None)))
            if init_expr is not None:
                val_reg = self._compile_expr_reg(init_expr, ctx)
                var_reg = ctx.variable(get_node_symbol(stmt))
                if val_reg != var_reg:
                    ctx.emit(Move(dst=var_reg, src=val_reg, origin=_span_of(stmt)))
            return

        if kind in ("AssignStmt", "AssignmentStmt", "Assignment", "Assign"):
            value_expr = getattr(stmt, "value", getattr(stmt, "expr", getattr(stmt, "right", None)))
            target_node = getattr(stmt, "target", getattr(stmt, "left", None))
            sym = get_node_symbol(stmt) or get_node_symbol(target_node)
            val_reg = self._compile_expr_reg(value_expr, ctx)
            var_reg = ctx.variable(sym)
            if val_reg != var_reg:
                ctx.emit(Move(dst=var_reg, src=val_reg, origin=_span_of(stmt)))
            return

        if kind == "ReturnStmt":
            val_expr = getattr(stmt, "value", getattr(stmt, "expr", None))
            if val_expr is not None:
                val_reg = self._compile_expr_reg(val_expr, ctx)
                ctx.emit(Return(value=val_reg, origin=_span_of(stmt)))
            else:
                ctx.emit(Return(value=None, origin=_span_of(stmt)))
            return

        raise ValueError(f"Comando desconhecido {kind} em {_span_of(stmt)}")

    def _require_choice(self, table: SelectionTable, node: Any, result_class: Any) -> Choice:
        choice = table.lookup(node, result_class)
        if choice is None:
            span = _span_of(node)
            raise ValueError(
                f"Sem cobertura para o nó {type(node).__name__} com classe {result_class} na posição {span}"
            )
        return choice

    def _compile_expr_reg(self, expr: Any, ctx: Context) -> VReg:
        table = self.selector.analyze(expr)
        choice = self._require_choice(table, expr, REG)
        return self._emit_choice(expr, choice, table, ctx)

    def _compile_expr_cond(
        self, expr: Any, ctx: Context, true_label: Label, false_label: Label
    ) -> None:
        table = self.selector.analyze(expr)
        choice = self._require_choice(table, expr, COND)
        self._emit_choice(expr, choice, table, ctx, true_label, false_label)

    def _compile_expr_effect(self, expr: Any, ctx: Context) -> None:
        table = self.selector.analyze(expr)
        choice = table.lookup(expr, EFFECT)
        if choice is not None:
            self._emit_choice(expr, choice, table, ctx)
            return
        choice = self._require_choice(table, expr, REG)
        self._emit_choice(expr, choice, table, ctx)

    def _emit_subproblem(
        self,
        sub: tuple[Any, Any],
        table: SelectionTable,
        ctx: Context,
        true_label: Label | None = None,
        false_label: Label | None = None,
    ) -> Any:
        sub_node, sub_class = sub
        sub_choice = self._require_choice(table, sub_node, sub_class)
        return self._emit_choice(sub_node, sub_choice, table, ctx, true_label, false_label)

    def _emit_choice(
        self,
        node: Any,
        choice: Choice,
        table: SelectionTable,
        ctx: Context,
        true_label: Label | None = None,
        false_label: Label | None = None,
    ) -> Any:
        action = getattr(self, choice.rule.action)
        return action(node, choice, table, ctx, true_label, false_label)

    def emit_literal_int_reg(self, node, choice, table, ctx, true_label=None, false_label=None) -> VReg:
        dst = ctx.temp()
        ctx.emit(Li(dst=dst, imm=_extract_signed_int(node), origin=_span_of(node)))
        return dst

    def emit_literal_bool_reg(self, node, choice, table, ctx, true_label=None, false_label=None) -> VReg:
        dst = ctx.temp()
        imm = 1 if getattr(node, "value", False) else 0
        ctx.emit(Li(dst=dst, imm=imm, origin=_span_of(node)))
        return dst

    def emit_ident_reg(self, node, choice, table, ctx, true_label=None, false_label=None) -> VReg:
        return ctx.variable(get_node_symbol(node))

    def emit_literal_imm12(self, node, choice, table, ctx, true_label=None, false_label=None) -> int:
        return _extract_signed_int(node)

    def emit_literal_pow2(self, node, choice, table, ctx, true_label=None, false_label=None) -> int:
        return _extract_pow2_shift(node)

    def emit_neg_reg(self, node, choice, table, ctx, true_label=None, false_label=None) -> VReg:
        src = self._emit_subproblem(choice.subproblems[0], table, ctx)
        dst = ctx.temp()
        ctx.emit(Neg(dst=dst, src=src, origin=_span_of(node)))
        return dst

    def _emit_bin_arith(self, instr_cls, node, choice, table, ctx) -> VReg:
        lhs = self._emit_subproblem(choice.subproblems[0], table, ctx)
        rhs = self._emit_subproblem(choice.subproblems[1], table, ctx)
        dst = ctx.temp()
        ctx.emit(instr_cls(dst=dst, lhs=lhs, rhs=rhs, origin=_span_of(node)))
        return dst

    def emit_add_reg(self, node, choice, table, ctx, true_label=None, false_label=None) -> VReg:
        return self._emit_bin_arith(Add, node, choice, table, ctx)

    def emit_sub_reg(self, node, choice, table, ctx, true_label=None, false_label=None) -> VReg:
        return self._emit_bin_arith(Sub, node, choice, table, ctx)

    def emit_mul_reg(self, node, choice, table, ctx, true_label=None, false_label=None) -> VReg:
        return self._emit_bin_arith(Mul, node, choice, table, ctx)

    def emit_div_reg(self, node, choice, table, ctx, true_label=None, false_label=None) -> VReg:
        return self._emit_bin_arith(Div, node, choice, table, ctx)

    def emit_rem_reg(self, node, choice, table, ctx, true_label=None, false_label=None) -> VReg:
        return self._emit_bin_arith(Rem, node, choice, table, ctx)

    def emit_add_imm_right(self, node, choice, table, ctx, true_label=None, false_label=None) -> VReg:
        src = self._emit_subproblem(choice.subproblems[0], table, ctx)
        imm = self._emit_subproblem(choice.subproblems[1], table, ctx)
        dst = ctx.temp()
        ctx.emit(Addi(dst=dst, src=src, imm=imm, origin=_span_of(node)))
        return dst

    def emit_add_imm_left(self, node, choice, table, ctx, true_label=None, false_label=None) -> VReg:
        imm = self._emit_subproblem(choice.subproblems[0], table, ctx)
        src = self._emit_subproblem(choice.subproblems[1], table, ctx)
        dst = ctx.temp()
        ctx.emit(Addi(dst=dst, src=src, imm=imm, origin=_span_of(node)))
        return dst

    def emit_sub_imm_right(self, node, choice, table, ctx, true_label=None, false_label=None) -> VReg:
        src = self._emit_subproblem(choice.subproblems[0], table, ctx)
        right_node = patterns.children(node)[1]
        imm = -_extract_signed_int(right_node)
        dst = ctx.temp()
        ctx.emit(Addi(dst=dst, src=src, imm=imm, origin=_span_of(node)))
        return dst

    def emit_mul_pow2_right(self, node, choice, table, ctx, true_label=None, false_label=None) -> VReg:
        src = self._emit_subproblem(choice.subproblems[0], table, ctx)
        shift = self._emit_subproblem(choice.subproblems[1], table, ctx)
        dst = ctx.temp()
        ctx.emit(Slli(dst=dst, src=src, shift=shift, origin=_span_of(node)))
        return dst

    def emit_mul_pow2_left(self, node, choice, table, ctx, true_label=None, false_label=None) -> VReg:
        shift = self._emit_subproblem(choice.subproblems[0], table, ctx)
        src = self._emit_subproblem(choice.subproblems[1], table, ctx)
        dst = ctx.temp()
        ctx.emit(Slli(dst=dst, src=src, shift=shift, origin=_span_of(node)))
        return dst

    def emit_lt_reg(self, node, choice, table, ctx, true_label=None, false_label=None) -> VReg:
        lhs = self._emit_subproblem(choice.subproblems[0], table, ctx)
        rhs = self._emit_subproblem(choice.subproblems[1], table, ctx)
        dst = ctx.temp()
        ctx.emit(Slt(dst=dst, lhs=lhs, rhs=rhs, origin=_span_of(node)))
        return dst

    def emit_gt_reg(self, node, choice, table, ctx, true_label=None, false_label=None) -> VReg:
        lhs = self._emit_subproblem(choice.subproblems[0], table, ctx)
        rhs = self._emit_subproblem(choice.subproblems[1], table, ctx)
        dst = ctx.temp()
        ctx.emit(Slt(dst=dst, lhs=rhs, rhs=lhs, origin=_span_of(node)))
        return dst

    def emit_le_reg(self, node, choice, table, ctx, true_label=None, false_label=None) -> VReg:
        lhs = self._emit_subproblem(choice.subproblems[0], table, ctx)
        rhs = self._emit_subproblem(choice.subproblems[1], table, ctx)
        dst = ctx.temp()
        ctx.emit(Slt(dst=dst, lhs=rhs, rhs=lhs, origin=_span_of(node)))
        ctx.emit(Seqz(dst=dst, src=dst, origin=_span_of(node)))
        return dst

    def emit_ge_reg(self, node, choice, table, ctx, true_label=None, false_label=None) -> VReg:
        lhs = self._emit_subproblem(choice.subproblems[0], table, ctx)
        rhs = self._emit_subproblem(choice.subproblems[1], table, ctx)
        dst = ctx.temp()
        ctx.emit(Slt(dst=dst, lhs=lhs, rhs=rhs, origin=_span_of(node)))
        ctx.emit(Seqz(dst=dst, src=dst, origin=_span_of(node)))
        return dst

    def emit_eq_reg(self, node, choice, table, ctx, true_label=None, false_label=None) -> VReg:
        lhs = self._emit_subproblem(choice.subproblems[0], table, ctx)
        rhs = self._emit_subproblem(choice.subproblems[1], table, ctx)
        dst = ctx.temp()
        ctx.emit(Xor(dst=dst, lhs=lhs, rhs=rhs, origin=_span_of(node)))
        ctx.emit(Seqz(dst=dst, src=dst, origin=_span_of(node)))
        return dst

    def emit_ne_reg(self, node, choice, table, ctx, true_label=None, false_label=None) -> VReg:
        lhs = self._emit_subproblem(choice.subproblems[0], table, ctx)
        rhs = self._emit_subproblem(choice.subproblems[1], table, ctx)
        dst = ctx.temp()
        ctx.emit(Xor(dst=dst, lhs=lhs, rhs=rhs, origin=_span_of(node)))
        ctx.emit(Snez(dst=dst, src=dst, origin=_span_of(node)))
        return dst

    def emit_lt_cond(self, node, choice, table, ctx, true_label, false_label) -> None:
        lhs = self._emit_subproblem(choice.subproblems[0], table, ctx)
        rhs = self._emit_subproblem(choice.subproblems[1], table, ctx)
        ctx.emit(Blt(lhs=lhs, rhs=rhs, target=true_label, origin=_span_of(node)))
        ctx.emit(Jump(target=false_label, origin=_span_of(node)))

    def emit_gt_cond(self, node, choice, table, ctx, true_label, false_label) -> None:
        lhs = self._emit_subproblem(choice.subproblems[0], table, ctx)
        rhs = self._emit_subproblem(choice.subproblems[1], table, ctx)
        ctx.emit(Blt(lhs=rhs, rhs=lhs, target=true_label, origin=_span_of(node)))
        ctx.emit(Jump(target=false_label, origin=_span_of(node)))

    def emit_le_cond(self, node, choice, table, ctx, true_label, false_label) -> None:
        lhs = self._emit_subproblem(choice.subproblems[0], table, ctx)
        rhs = self._emit_subproblem(choice.subproblems[1], table, ctx)
        ctx.emit(Bge(lhs=rhs, rhs=lhs, target=true_label, origin=_span_of(node)))
        ctx.emit(Jump(target=false_label, origin=_span_of(node)))

    def emit_ge_cond(self, node, choice, table, ctx, true_label, false_label) -> None:
        lhs = self._emit_subproblem(choice.subproblems[0], table, ctx)
        rhs = self._emit_subproblem(choice.subproblems[1], table, ctx)
        ctx.emit(Bge(lhs=lhs, rhs=rhs, target=true_label, origin=_span_of(node)))
        ctx.emit(Jump(target=false_label, origin=_span_of(node)))

    def emit_eq_cond(self, node, choice, table, ctx, true_label, false_label) -> None:
        lhs = self._emit_subproblem(choice.subproblems[0], table, ctx)
        rhs = self._emit_subproblem(choice.subproblems[1], table, ctx)
        ctx.emit(Beq(lhs=lhs, rhs=rhs, target=true_label, origin=_span_of(node)))
        ctx.emit(Jump(target=false_label, origin=_span_of(node)))

    def emit_ne_cond(self, node, choice, table, ctx, true_label, false_label) -> None:
        lhs = self._emit_subproblem(choice.subproblems[0], table, ctx)
        rhs = self._emit_subproblem(choice.subproblems[1], table, ctx)
        ctx.emit(Bne(lhs=lhs, rhs=rhs, target=true_label, origin=_span_of(node)))
        ctx.emit(Jump(target=false_label, origin=_span_of(node)))

    def emit_not_reg(self, node, choice, table, ctx, true_label=None, false_label=None) -> VReg:
        src = self._emit_subproblem(choice.subproblems[0], table, ctx)
        dst = ctx.temp()
        ctx.emit(Seqz(dst=dst, src=src, origin=_span_of(node)))
        return dst

    def emit_not_cond(self, node, choice, table, ctx, true_label, false_label) -> None:
        self._emit_subproblem(choice.subproblems[0], table, ctx, false_label, true_label)

    def emit_and_cond(self, node, choice, table, ctx, true_label, false_label) -> None:
        mid_label = ctx.label()
        self._emit_subproblem(choice.subproblems[0], table, ctx, mid_label, false_label)
        ctx.emit(LabelInstr(label=mid_label))
        self._emit_subproblem(choice.subproblems[1], table, ctx, true_label, false_label)

    def emit_or_cond(self, node, choice, table, ctx, true_label, false_label) -> None:
        mid_label = ctx.label()
        self._emit_subproblem(choice.subproblems[0], table, ctx, true_label, mid_label)
        ctx.emit(LabelInstr(label=mid_label))
        self._emit_subproblem(choice.subproblems[1], table, ctx, true_label, false_label)

    def emit_and_reg(self, node, choice, table, ctx, true_label=None, false_label=None) -> VReg:
        mid_label = ctx.label()
        t_label = ctx.label()
        f_label = ctx.label()
        end_label = ctx.label()
        dst = ctx.temp()

        self._emit_subproblem(choice.subproblems[0], table, ctx, mid_label, f_label)
        ctx.emit(LabelInstr(label=mid_label))
        self._emit_subproblem(choice.subproblems[1], table, ctx, t_label, f_label)
        ctx.emit(LabelInstr(label=t_label))
        ctx.emit(Li(dst=dst, imm=1, origin=_span_of(node)))
        ctx.emit(Jump(target=end_label, origin=_span_of(node)))
        ctx.emit(LabelInstr(label=f_label))
        ctx.emit(Li(dst=dst, imm=0, origin=_span_of(node)))
        ctx.emit(LabelInstr(label=end_label))
        return dst

    def emit_or_reg(self, node, choice, table, ctx, true_label=None, false_label=None) -> VReg:
        mid_label = ctx.label()
        t_label = ctx.label()
        f_label = ctx.label()
        end_label = ctx.label()
        dst = ctx.temp()

        self._emit_subproblem(choice.subproblems[0], table, ctx, t_label, mid_label)
        ctx.emit(LabelInstr(label=mid_label))
        self._emit_subproblem(choice.subproblems[1], table, ctx, t_label, f_label)
        ctx.emit(LabelInstr(label=t_label))
        ctx.emit(Li(dst=dst, imm=1, origin=_span_of(node)))
        ctx.emit(Jump(target=end_label, origin=_span_of(node)))
        ctx.emit(LabelInstr(label=f_label))
        ctx.emit(Li(dst=dst, imm=0, origin=_span_of(node)))
        ctx.emit(LabelInstr(label=end_label))
        return dst

    def emit_literal_bool_cond(self, node, choice, table, ctx, true_label, false_label) -> None:
        target = true_label if getattr(node, "value", False) else false_label
        ctx.emit(Jump(target=target, origin=_span_of(node)))

    def emit_ident_bool_cond(self, node, choice, table, ctx, true_label, false_label) -> None:
        vreg = ctx.variable(get_node_symbol(node))
        ctx.emit(Bnez(src=vreg, target=true_label, origin=_span_of(node)))
        ctx.emit(Jump(target=false_label, origin=_span_of(node)))

    def emit_call_reg(self, node, choice, table, ctx, true_label=None, false_label=None) -> VReg:
        arg_regs = [self._emit_subproblem(sub, table, ctx) for sub in choice.subproblems]
        dst = ctx.temp()
        ctx.emit(Call(dst=dst, function=get_node_symbol(node), args=arg_regs, origin=_span_of(node)))
        return dst

    def emit_call_effect(self, node, choice, table, ctx, true_label=None, false_label=None) -> None:
        arg_regs = [self._emit_subproblem(sub, table, ctx) for sub in choice.subproblems]
        ctx.emit(Call(dst=None, function=get_node_symbol(node), args=arg_regs, origin=_span_of(node)))

    def emit_call_bool_cond(self, node, choice, table, ctx, true_label, false_label) -> None:
        arg_regs = [self._emit_subproblem(sub, table, ctx) for sub in choice.subproblems]
        dst = ctx.temp()
        ctx.emit(Call(dst=dst, function=get_node_symbol(node), args=arg_regs, origin=_span_of(node)))
        ctx.emit(Bnez(src=dst, target=true_label, origin=_span_of(node)))
        ctx.emit(Jump(target=false_label, origin=_span_of(node)))