"""入出力層: メッシュ(GLB/glTF/OBJ)とスタンドアロン画像の読み書き。

trimesh/Pillow への依存はこのパッケージ配下に閉じ込める(横断規約の依存方向)。
公開 API は `load_mesh`/`save_mesh` の2関数のみ(横断規約の「公開API 5関数」契約
— オーケストレーター裁定)。画像 I/O(`load_image`/`save_image`)はここで
re-export せず、`atlasmith.io.image` モジュール経由の内部抽象に留める
(bake 等からは `from atlasmith.io.image import load_image` の形で利用する)。
"""

from atlasmith.io.mesh import load_mesh, save_mesh

__all__ = ["load_mesh", "save_mesh"]
