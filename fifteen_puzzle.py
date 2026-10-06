#!/usr/bin/env python3
"""fifteen-puzzle: 4x4 滑块拼图（15-puzzle），带 A* 求解器。

规则：把 1-15 按顺序排好，0 为空格。每次把空格相邻的数字滑入空格。
"""
from __future__ import annotations

import argparse
import heapq
import random
import sys
import time
from copy import deepcopy

N = 4
GOAL = tuple(list(range(1, 16)) + [0])
DIRS = {"w": (-1, 0), "s": (1, 0), "a": (0, -1), "d": (0, 1)}
DIR_NAME = {"w": "空格上移", "s": "空格下移", "a": "空格左移", "d": "空格右移"}

# 曼哈顿距离表：tile -> pos -> dist
_MANHATTAN = {}
for tile in range(1, 16):
    gr, gc = divmod(tile - 1, N)
    for pos in range(16):
        r, c = divmod(pos, N)
        _MANHATTAN[(tile, pos)] = abs(r - gr) + abs(c - gc)


class IllegalMove(Exception):
    pass


def rc(pos):
    return divmod(pos, N)


def pos_of(r, c):
    return r * N + c


def inversions(board) -> int:
    tiles = [t for t in board if t != 0]
    inv = 0
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            if tiles[i] > tiles[j]:
                inv += 1
    return inv


def is_solvable(board) -> bool:
    """4x4：逆序数 + 空格所在行（从底往上数，1 起）为奇数则可解。"""
    blank_row_from_bottom = N - rc(board.index(0))[0]
    return (inversions(board) + blank_row_from_bottom) % 2 == 1


def is_solved(board) -> bool:
    return tuple(board) == GOAL


def blank_neighbors(board):
    """空格可走的四个方向（返回方向字母）。"""
    b = board.index(0)
    r, c = rc(b)
    out = []
    for key, (dr, dc) in DIRS.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < N and 0 <= nc < N:
            out.append(key)
    return out


def apply_move(board, key):
    """把空格往 key 方向移一步（非法抛 IllegalMove），返回新 board（list）。"""
    if key not in DIRS:
        raise IllegalMove(f"未知方向: {key}")
    b = board.index(0)
    r, c = rc(b)
    dr, dc = DIRS[key]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < N and 0 <= nc < N):
        raise IllegalMove(f"空格不能再往 {DIR_NAME[key]}")
    nb = deepcopy(list(board))
    t = pos_of(nr, nc)
    nb[b], nb[t] = nb[t], nb[b]
    return nb


def scramble(depth: int, rng: random.Random) -> tuple:
    """从目标状态随机走 depth 步（保证可解）。"""
    board = list(GOAL)
    prev = None
    for _ in range(depth):
        choices = [k for k in blank_neighbors(board)
                   if not (prev and {"w": "s", "s": "w", "a": "d", "d": "a"}[prev] == k)]
        key = rng.choice(choices)
        board = apply_move(board, key)
        prev = key
    return tuple(board)


def manhattan(board) -> int:
    return sum(_MANHATTAN[(t, p)] for p, t in enumerate(board) if t != 0)


def solve(start, node_limit: int = 3_000_000):
    """A*（曼哈顿启发式）求最优解。返回 (moves列表, 扩展节点数, 用时秒)；超限返回 None。"""
    start = tuple(start)
    if start == GOAL:
        return [], 0, 0.0
    t0 = time.perf_counter()
    open_heap = [(manhattan(start), 0, start)]
    came_from = {start: (None, None)}
    g_score = {start: 0}
    expanded = 0
    while open_heap:
        f, g, cur = heapq.heappop(open_heap)
        if g != g_score.get(cur):
            continue
        if cur == GOAL:
            path = []
            node = cur
            while came_from[node][0] is not None:
                node, mv = came_from[node]
                path.append(mv)
            path.reverse()
            return path, expanded, time.perf_counter() - t0
        expanded += 1
        if expanded > node_limit:
            return None
        b = cur.index(0)
        r, c = rc(b)
        for key, (dr, dc) in DIRS.items():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < N and 0 <= nc < N):
                continue
            lst = list(cur)
            t = pos_of(nr, nc)
            lst[b], lst[t] = lst[t], lst[b]
            nxt = tuple(lst)
            ng = g + 1
            if ng < g_score.get(nxt, float("inf")):
                g_score[nxt] = ng
                came_from[nxt] = (cur, key)
                heapq.heappush(open_heap, (ng + manhattan(nxt), ng, nxt))
    return None


def render(board) -> str:
    lines = []
    for r in range(N):
        row = []
        for c in range(N):
            t = board[pos_of(r, c)]
            row.append("   " if t == 0 else f"{t:>3}")
        lines.append(" ".join(row))
    return "\n".join(lines)


def play_interactive(seed=None):
    rng = random.Random(seed)
    board = scramble(40, rng)
    moves = 0
    print("十五拼图：把数字滑成 1-15 顺序。w/a/s/d 移动空格，q 退出。")
    while True:
        print()
        print(render(board))
        print(f"已走 {moves} 步")
        if is_solved(board):
            print(f"拼好啦！共 {moves} 步。")
            return
        try:
            cmd = input("走法 [w/a/s/d/q]: ").strip().lower()
        except EOFError:
            print("\n退出。")
            return
        if cmd == "q":
            print("退出。")
            return
        try:
            board = apply_move(board, cmd)
            moves += 1
        except IllegalMove as e:
            print(f"非法走法：{e}")


def cmd_auto(args):
    rng = random.Random(args.seed)
    total_len = 0
    total_nodes = 0
    total_time = 0.0
    ok = 0
    for i in range(args.games):
        board = scramble(args.depth, rng)
        res = solve(board)
        if res is None:
            print(f"第 {i+1}/{args.games} 局：求解超限（>{3_000_000} 节点），跳过")
            continue
        path, nodes, secs = res
        total_len += len(path)
        total_nodes += nodes
        total_time += secs
        ok += 1
        if args.verbose:
            print(f"第 {i+1}/{args.games} 局：最优解 {len(path)} 步，扩展 {nodes} 节点，{secs:.2f}s")
    print(f"自动演示结束：共 {args.games} 局，求解成功 {ok} 局")
    if ok:
        print(f"平均最优步数 {total_len/ok:.1f}，平均扩展节点 {total_nodes/ok:.0f}，平均用时 {total_time/ok:.2f}s")


def cmd_solve(args):
    rng = random.Random(args.seed)
    board = scramble(args.depth, rng)
    print("初始局面：")
    print(render(board))
    print(f"可解性检查：{'可解' if is_solvable(board) else '不可解（不应发生）'}")
    res = solve(board)
    if res is None:
        print("求解超限，未找到解。")
        return
    path, nodes, secs = res
    print(f"最优解：{len(path)} 步（{''.join(path)}），扩展 {nodes} 节点，用时 {secs:.2f}s")
    if args.show:
        b = list(board)
        for i, mv in enumerate(path, 1):
            b = apply_move(b, mv)
            print(f"\n第 {i} 步（{DIR_NAME[mv]}）：")
            print(render(b))


def main(argv=None):
    ap = argparse.ArgumentParser(description="十五拼图（15-puzzle）+ A* 求解器")
    ap.add_argument("--seed", type=int, default=None, help="随机种子")
    sub = ap.add_subparsers(dest="cmd")
    pa = sub.add_parser("auto", help="自动演示：打乱 + 求解")
    pa.add_argument("--seed", type=int, default=None, help="随机种子")
    pa.add_argument("--games", type=int, default=5)
    pa.add_argument("--depth", type=int, default=40, help="每局打乱步数")
    pa.add_argument("--verbose", action="store_true")
    ps = sub.add_parser("solve", help="打乱一局并求解")
    ps.add_argument("--seed", type=int, default=None, help="随机种子")
    ps.add_argument("--depth", type=int, default=40)
    ps.add_argument("--show", action="store_true", help="逐步展示解法")
    args = ap.parse_args(argv)
    if args.cmd == "auto":
        cmd_auto(args)
    elif args.cmd == "solve":
        cmd_solve(args)
    else:
        if not sys.stdin.isatty():
            print("交互模式需要终端；请用 auto 或 solve 子命令。", file=sys.stderr)
            return 2
        play_interactive(args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
