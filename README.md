# Atlasmith

AI生成3Dモデル(Tripo/Meshy/Hunyuan3D 等)の乱雑な UV を部位単位のアイランドへ再編成し、
既存テクスチャを新UVへ焼き直す Python 製 CLI ツール。パイプラインは5段(①部位分割 ②シーム決定
③平面展開 ④パッキング ⑤テクスチャ焼き直し)。現状は Phase 0(GLB/glTF/OBJ 入出力)実装済み、
Phase 1(⑤テクスチャ焼き直し)を実装中。

## インストール

[uv](https://docs.astral.sh/uv/) 前提。Python 3.12(開発 pin)は `uv sync` 実行時に uv が
自動で導入する(手動インストール不要)。

```
uv sync
```

## 使い方(現状)

```
uv run atlasmith input.glb -o output.glb
```

**現段階は load→save のラウンドトリップのみ**(読み込んだメッシュをそのまま書き出す)。
UV 再展開+テクスチャ焼き直しの結線は Phase 1 完了時(Step 1-3)にこの節を更新する。

## 制約(承認済みの正式制約)

- 対応範囲は単一メッシュ・単一マテリアル・単一UVセットに限定する。
- normal map は転写するが警告付き(UV 変更でタンジェント空間の基底が変わるため、
  照明的正しさは保証しない)。
- OBJ 書き出しは basecolor のみ保持する(trimesh の `SimpleMaterial` の制約により
  normal・metallic_roughness は落ちる)。

## 開発

```
uv run ruff format --check .    # フォーマット
uv run ruff check .             # リント
uv run pytest                   # テスト
```

## ライセンス

Apache-2.0
