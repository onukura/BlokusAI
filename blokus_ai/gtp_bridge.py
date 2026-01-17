"""Pentobi GTP エンジンとの通信ブリッジ。

このモジュールはPentobiのGTPプロトコルを使って、BlokusAIとPentobiエンジンを統合します。
"""

from __future__ import annotations

import subprocess
import sys
from typing import List, Tuple

from blokus_ai.engine import Engine, Move
from blokus_ai.state import GameState


class PentobiGTPEngine:
    """PentobiエンジンとのGTPプロトコル通信を管理するクラス。

    Attributes:
        process: pentobi_gtpのサブプロセス
        game_variant: ゲームバリアント（"duo", "classic"など）
        level: エンジンレベル（1-8）
        board_size: ボードサイズ（Duo=14, Classic=20）
    """

    def __init__(
        self,
        pentobi_path: str = "pentobi_gtp",
        game_variant: str = "duo",
        level: int = 5,
        quiet: bool = True,
    ):
        """PentobiGTPEngineを初期化する。

        Args:
            pentobi_path: pentobi_gtp実行ファイルのパス
            game_variant: ゲームバリアント（"duo", "classic"など）
            level: エンジンレベル（1-8、8が最強）
            quiet: Trueの場合、Pentobiのログ出力を抑制
        """
        self.game_variant = game_variant
        self.level = level
        self.board_size = 14 if game_variant == "duo" else 20

        # pentobi_gtpプロセスを起動
        cmd = [pentobi_path, "--game", game_variant, "--level", str(level)]
        if quiet:
            cmd.append("--quiet")

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # ゲームを初期化
        self.send_command("clear_board")

    def send_command(self, command: str) -> str:
        """GTPコマンドを送信して応答を取得する。

        Args:
            command: GTPコマンド文字列

        Returns:
            Pentobiからの応答文字列（成功時）

        Raises:
            RuntimeError: コマンド実行に失敗した場合
        """
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("Pentobi process not initialized")

        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

        # 応答を読み取る（"= "で始まる行または"? "でエラー）
        response_lines = []
        while True:
            line = self.process.stdout.readline().strip()
            if line.startswith("="):
                # 成功応答
                response_lines.append(line[2:])  # "= "を除去
                # 空行まで読み続ける
                while True:
                    line = self.process.stdout.readline().strip()
                    if not line:
                        break
                    response_lines.append(line)
                break
            elif line.startswith("?"):
                # エラー応答
                error_msg = line[2:]  # "? "を除去
                raise RuntimeError(f"GTP command failed: {command}\nError: {error_msg}")

        return "\n".join(response_lines)

    def clear_board(self):
        """ボードをクリアして新規ゲームを開始する。"""
        self.send_command("clear_board")

    def play_move(self, player: int, move_coords: str):
        """手を適用する。

        Args:
            player: プレイヤーID（0または1）
            move_coords: Pentobi形式の座標文字列（例: "f9,e10,f10,g10,f11"）
        """
        color = "B" if player == 0 else "W"
        self.send_command(f"play {color} {move_coords}")

    def genmove(self, player: int) -> str:
        """エンジンに手を生成させる。

        Args:
            player: プレイヤーID（0または1）

        Returns:
            Pentobi形式の座標文字列（例: "f9,e10,f10,g10,f11"）、
            またはパスの場合は"pass"
        """
        color = "B" if player == 0 else "W"
        response = self.send_command(f"genmove {color}")
        return response.strip()

    def quit(self):
        """Pentobiエンジンを終了する。"""
        try:
            self.send_command("quit")
        except Exception:
            pass
        finally:
            if self.process:
                self.process.terminate()
                self.process.wait()

    def __del__(self):
        """デストラクタでプロセスを確実に終了する。"""
        self.quit()


def blokus_coord_to_pentobi(coord: Tuple[int, int], board_size: int = 14) -> str:
    """BlokusAIの座標をPentobi形式に変換する。

    BlokusAI: (y, x)の0-indexed、左上が(0, 0)
    Pentobi: "a1"形式、左下がa1

    Args:
        coord: BlokusAIの座標（y, x）
        board_size: ボードサイズ（デフォルト14）

    Returns:
        Pentobi形式の座標文字列（例: "a1"）
    """
    y, x = coord
    # BlokusAIのy座標を反転（左上→左下基準）
    pentobi_row = board_size - y
    # x座標をa,b,c...に変換
    pentobi_col = chr(ord("a") + x)
    return f"{pentobi_col}{pentobi_row}"


def pentobi_coord_to_blokus(coord_str: str, board_size: int = 14) -> Tuple[int, int]:
    """Pentobi形式の座標をBlokusAI形式に変換する。

    Args:
        coord_str: Pentobi形式の座標文字列（例: "a1"）
        board_size: ボードサイズ（デフォルト14）

    Returns:
        BlokusAIの座標（y, x）
    """
    col_str = coord_str[0]
    row_str = coord_str[1:]

    x = ord(col_str) - ord("a")
    pentobi_row = int(row_str)
    y = board_size - pentobi_row

    return (y, x)


def blokus_move_to_pentobi(move: Move, board_size: int = 14) -> str:
    """BlokusAIのMoveオブジェクトをPentobi形式に変換する。

    Args:
        move: BlokusAIのMove オブジェクト
        board_size: ボードサイズ（デフォルト14）

    Returns:
        Pentobi形式の座標文字列（例: "f9,e10,f10,g10,f11"）
    """
    pentobi_coords = [blokus_coord_to_pentobi(cell, board_size) for cell in move.cells]
    return ",".join(pentobi_coords)


def pentobi_move_to_blokus_index(
    pentobi_move: str, legal_moves: List[Move], board_size: int = 14
) -> int:
    """Pentobi形式の手をBlokusAIの合法手インデックスに変換する。

    Args:
        pentobi_move: Pentobi形式の座標文字列（例: "f9,e10,f10,g10,f11"）
        legal_moves: BlokusAIの合法手リスト
        board_size: ボードサイズ（デフォルト14）

    Returns:
        合法手リスト内のインデックス

    Raises:
        ValueError: 対応する合法手が見つからない場合
    """
    if pentobi_move.lower() == "pass":
        return -1

    # Pentobi形式の座標をBlokusAI形式に変換
    pentobi_coords = pentobi_move.split(",")
    blokus_coords = set(
        pentobi_coord_to_blokus(coord.strip(), board_size) for coord in pentobi_coords
    )

    # 合法手の中から一致するものを探す
    for idx, move in enumerate(legal_moves):
        if set(move.cells) == blokus_coords:
            return idx

    raise ValueError(
        f"Pentobi move '{pentobi_move}' does not match any legal move in BlokusAI"
    )


def state_to_gtp_commands(state: GameState, engine: Engine) -> List[str]:
    """GameStateをGTPコマンドのシーケンスに変換する。

    ゲームの現在の状態をPentobiエンジンに再現するためのコマンド列を生成します。

    Args:
        state: 現在のGameState
        engine: Engineインスタンス

    Returns:
        GTPコマンド文字列のリスト

    Note:
        現在の実装は簡易版で、ボードの状態を直接再現することはできません。
        実際の使用では、ゲームの最初から全ての手を記録して再現する必要があります。
    """
    commands = ["clear_board"]
    # TODO: 実際の手の履歴を記録して再現する実装が必要
    # 現時点では新規ゲームとして扱う
    return commands
