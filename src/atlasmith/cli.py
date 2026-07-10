"""Atlasmith CLI エントリポイント。現状は --help のみのスタブ。"""

import argparse


def main(argv: list[str] | None = None) -> int:
    """CLI エントリポイント。引数の結線は次ステップで行う。"""
    parser = argparse.ArgumentParser(prog="atlasmith")
    parser.parse_args(argv)
    return 0
