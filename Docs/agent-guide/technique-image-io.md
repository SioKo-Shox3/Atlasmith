# 技術選定記録: image-io

日付: 2026-07-10 ／ 状態: 決定済み ／ 決定者: ユーザー承認(2026-07-10、AskUserQuestion で「Pillow (Recommended)」を選択。品質契約・正式制約の承認を含む)

## 課題定義
- 解こうとしている問題: テクスチャ焼き直し(bake)ツールが読み書きする画像フォーマット(basecolor 等の PNG/JPEG)の I/O ライブラリを選定する。
- 制約(技術・規約・予算): Apache-2.0 と非互換のライセンスは不可(絶対規則3)。依存追加はユーザー承認が必要(絶対規則1)。Windows cp312 wheel が入手できること(technique-package-manager.md で確定した環境)。Phase 0/1 の正式制約(単一メッシュ・単一マテリアル・単一UVセット)のスコープを超える機能(EXR・16bit float・UDIM 等)への対応は不要。
- 成功基準(何が満たされれば成功か): (1) basecolor 等の一般的な 8bit PNG/JPEG テクスチャの読み書きができる。(2) trimesh のテクスチャデコードとも整合し、追加の変換コストが発生しない。(3) 将来 EXR/16bit float 等の需要が出た場合に、実装全体を作り直さずバックエンドを差し替えられる設計になっている。

## 候補

| 候補 | 概要 | 主な根拠 | 主な懸念 |
|---|---|---|---|
| Pillow(採用) | 定番の Python 画像 I/O ライブラリ | 軽量、trimesh のテクスチャデコードにも必要(trimesh[easy] 相当)、MIT-CMU ライセンス | EXR・16bit float 等の高度なフォーマットは非対応 |
| OpenImageIO(OIIO) | VFX 業界標準の画像 I/O ライブラリ、ASWF(Academy Software Foundation)公式配布 | EXR・16bit float・UDIM 等の高度なフォーマットに対応、Windows wheel(cp310〜cp313)が実用段階 | Phase 0/1 の要件(8bit PNG/JPEG のみ)には過剰。依存サイズ・導入コストが Pillow より大きい |
| 何もしない(画像 I/O ライブラリなし) | 依存を追加せず、標準ライブラリのみで画像を扱う | 依存追加の承認・ライセンス確認が不要 | Python 標準ライブラリには PNG/JPEG のデコード・エンコードを行う実用的な手段がなく、テクスチャ焼き直しツールとして成立しない。不採用。 |

## 証拠等級付き比較

| 論点 | Pillow(採用) | OpenImageIO | 何もしない |
|---|---|---|---|
| ライセンス | MIT-CMU【外部: PyPI Pillow 12.3.0、2026-07-01 参照】、Apache-2.0 互換 | ASWF 配布、BSD-3-Clause 系【外部: https://pypi.org/project/OpenImageIO/、2026-07-10 参照】、Apache-2.0 互換 | 該当なし |
| Windows wheel 対応(cp312) | 公式に cp312 wheel を配布【外部: PyPI Pillow ページ、2026-07-10 参照】 | ASWF 公式 Windows wheel(cp310〜cp313)が実用段階【外部: https://pypi.org/project/OpenImageIO/】→ 3.12 は圏内 | 該当なし |
| Phase 0/1 要件との整合 | basecolor 等の 8bit PNG/JPEG テクスチャの読み書きに直接対応、trimesh のテクスチャデコード要件(trimesh[easy] 相当)も満たす【外部: trimesh ドキュメントの optional dependency 記載、一般公知情報】 | EXR/16bit float/UDIM 対応は Phase 0/1 の正式制約(単一メッシュ・単一マテリアル・単一UVセット、8bit 前提の品質契約)には不要な過剰機能【推測: 計画書の品質契約(data_range=1.0 float32・8bit 前提)との突合】 | 画像 I/O ができずテクスチャ焼き直しツールとして不成立【推測】 |
| 導入・保守コスト | 軽量で広く使われる定番ライブラリ、依存グラフが小さい【推測: 一般的な Python エコシステムの知見】 | VFX 業界標準だが依存グラフ・バイナリサイズが Pillow より大きい【推測】 | ゼロだが目的を達成できない |

## プロトタイプコスト
- 本命候補を安く検証する方法: Step 0-3 で `uv run python -c "import trimesh, numpy, xatlas, PIL"` を実行し、Pillow(`PIL`)の import が例外なく成功するかを実測する。
- 判定基準: import が成功すれば合格。将来 EXR/16bit float/UDIM の要望が出た場合は、`src/atlasmith/io/image.py` の薄い抽象層(load_image/save_image)を介して OpenImageIO 等のバックエンドを追加選定・差し替える。この差し替えコストの低さ自体が設計上の判定基準である。

## Advisor往復
- 相談日時・方向・要旨: 2026-07-10、オーケストレーターが画像 I/O ライブラリとして Pillow を含む計画をユーザーへ提示し、AskUserQuestion で承認を得た。パートナーAI(非メイン側)による計画二次レビューでは、bake モジュールが特定の画像ライブラリに密結合しないよう抽象層を設ける設計(`src/atlasmith/io/image.py` が load_image/save_image を提供し、bake/ は numpy のみに依存して PIL を import しない)が推奨され、計画に反映済み。反例として「最初から OpenImageIO を採用し EXR 等将来要件を先取りできないか」を検討したが、Phase 0/1 のスコープ(8bit PNG/JPEG のみ)には過剰であり Pillow で十分と判断した。

## 決定記録
- 採用: Pillow 12.3.0(MIT-CMU)。理由: 軽量で trimesh のテクスチャデコードにも必要(trimesh[easy] 相当)であり、Phase 0/1 の要件(8bit PNG/JPEG の読み書き)を過不足なく満たすため。設計上の補償として `src/atlasmith/io/image.py` を薄い抽象層(load_image/save_image)にしバックエンド差し替え可能にする。bake/ は PIL を import せず numpy のみに依存する。
- 却下:
  - OpenImageIO: 今回導入しない。理由は EXR/16bit float 対応が不要な Phase 0/1 には過剰であり、依存グラフ・導入コストが Pillow より大きいため。EXR/16bit float/UDIM 対応の要望が出た時点で追加選定する。
  - 何もしない(画像 I/O ライブラリなし): Python 標準ライブラリのみでは PNG/JPEG の実用的なデコード・エンコード手段がなく、テクスチャ焼き直しツールとして成立しないため不採用。
- **再評価トリガー**: (1) EXR 対応の要望が出た時点、(2) 16bit float テクスチャ対応の要望が出た時点、(3) UDIM 対応の要望が出た時点 — これらいずれかが発生した時点で OpenImageIO を追加選定する。
