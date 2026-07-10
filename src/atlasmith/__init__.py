"""Atlasmith: mesh texture rebaking toolkit (UV re-unwrap, atlas packing, bake)."""

from importlib.metadata import version

# インストール済みメタデータを唯一の情報源にする(pyproject.toml と二重管理しない)。
# 未インストール・メタデータ破損時は import 時点で例外を送出させ、隠さず顕在化させる。
__version__ = version("atlasmith")
