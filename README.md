# Atlasmith

AI生成3Dモデル(Tripo/Meshy/Hunyuan3D 等)の乱雑な UV を部位単位のアイランドへ再編成し、
既存テクスチャを新UVへ焼き直す Python 製 CLI ツール。パイプラインは5段(①部位分割 ②シーム決定
③平面展開 ④パッキング ⑤テクスチャ焼き直し)。現状は Phase 0(GLB/glTF/OBJ 入出力)+
Phase 1(⑤テクスチャ焼き直し: 読込→UV再展開→焼き直し→書出)実装済み。Phase 2 以降
(①〜④の本実装)は未着手。

## インストール

[uv](https://docs.astral.sh/uv/) 前提。Python 3.12(開発 pin)は `uv sync` 実行時に uv が
自動で導入する(手動インストール不要)。

```
uv sync
```

## 使い方

```
uv sync
uv run python examples/make_demo_assets.py
uv run atlasmith examples/demo.glb -o examples/demo_repacked.glb --padding 8
```

1行目で依存を導入し、2行目でテクスチャ付きデモメッシュ `examples/demo.glb` を
ローカル生成し、3行目でそれを読み込んで UV を再展開しテクスチャを焼き直した
`examples/demo_repacked.glb` を書き出す(読込→UV再展開→テクスチャ焼き直し→書出の
一気通貫パイプライン)。`--padding`(既定8)はチャート間パディング兼ガター膨張、
`--texture-size`(既定1024)は出力テクスチャの一辺(テクセル)。

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
