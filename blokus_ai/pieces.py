from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

Cell = Tuple[int, int]


@dataclass(frozen=True)
class PieceVariant:
    cells: Tuple[Cell, ...]

    @property
    def size(self) -> int:
        return len(self.cells)


@dataclass(frozen=True)
class Piece:
    name: str
    variants: Tuple[PieceVariant, ...]


def _parse_pattern(pattern: str) -> List[Cell]:
    lines = [line.rstrip() for line in pattern.splitlines() if line.strip() != ""]
    cells: List[Cell] = []
    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            if ch == "X":
                cells.append((x, y))
    return cells


def _normalize(cells: Iterable[Cell]) -> Tuple[Cell, ...]:
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    min_x = min(xs)
    min_y = min(ys)
    normalized = sorted((x - min_x, y - min_y) for x, y in cells)
    return tuple(normalized)


def _rotate(cells: Iterable[Cell]) -> Tuple[Cell, ...]:
    rotated = [(y, -x) for x, y in cells]
    return _normalize(rotated)


def _reflect(cells: Iterable[Cell]) -> Tuple[Cell, ...]:
    reflected = [(-x, y) for x, y in cells]
    return _normalize(reflected)


def _variants_for_pattern(pattern: str) -> Tuple[PieceVariant, ...]:
    base = _normalize(_parse_pattern(pattern))
    variants = set()
    current = base
    for _ in range(4):
        variants.add(current)
        variants.add(_reflect(current))
        current = _rotate(current)
    return tuple(PieceVariant(cells=variant) for variant in sorted(variants))


def build_blokus_pieces() -> Tuple[Piece, ...]:
    patterns = [
        ("mono", "X"),
        ("domino", "XX"),
        ("tri_i", "XXX"),
        ("tri_l", "XX\nX"),
        ("tet_i", "XXXX"),
        ("tet_o", "XX\nXX"),
        ("tet_t", "XXX\n.X."),
        ("tet_l", "XXX\nX.."),
        ("tet_s", "XX.\n.XX"),
        ("pent_f", ".XX\nXX.\n.X."),
        ("pent_i", "XXXXX"),
        ("pent_l", "XXXX\nX"),
        ("pent_n", "XXX\n..XX"),
        ("pent_p", "XX\nXX\nX."),
        ("pent_t", "XXX\n.X.\n.X."),
        ("pent_u", "X.X\nXXX"),
        ("pent_v", "X..\nX..\nXXX"),
        ("pent_w", "X..\nXX.\n.XX"),
        ("pent_x", ".X.\nXXX\n.X."),
        ("pent_y", "XXXX\n..X."),
        ("pent_z", "XX.\n.XX\n..X"),
    ]
    pieces = [
        Piece(name=name, variants=_variants_for_pattern(pattern))
        for name, pattern in patterns
    ]
    return tuple(pieces)


PIECES = build_blokus_pieces()
