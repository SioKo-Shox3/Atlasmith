# 技術選定記録: ML部位分割バックエンド(①部位分割)

日付: 2026-07-26(調査)/ 2026-07-27(決定) ／ 状態: **決定済み** ／
決定者: ユーザー承認「**調査結果見て判断します。SAM2 自前実装に移行しましょう**」(2026-07-27)

## 課題定義

- **解こうとしている問題**: AI生成3Dモデル(Tripo/Meshy/Hunyuan3D 等、マーチングキューブ由来で
  滑らか・エッジフロー無し)を**意味的な部位**(頭・胴・腕 等)に分割する。Phase 2 の①段。
- **幾何学的手法では不十分であることの根拠(最重要)**: 二面角クラスタリング+連結成分は
  「鋭い折れ目」と「別シェル」しか切れず、**滑らかな有機的キャラクターでは P=1 に退化する**。
  ユーザーが自身で幾何アプローチを試し限界を確認済み(2026-07-26 発言:「幾何的なアプローチは
  私も試してみてかなり限界を感じた」)。この実体験が本調査の起点。
- **制約**: Atlasmith は Apache-2.0(絶対規則3: 非互換依存を持ち込まない)。Windows 開発機
  (RTX 4080 / VRAM 16GB【実測】、CUDA Toolkit 11.2/11.3【実測】、VS Build Tools 2026【実測】、
  WSL ディストリ未導入【実測】)。ML は optional 依存とし、非搭載でも幾何フォールバックで動く。
- **成功基準**: (a) コード・重み・**学習データセット**の3つすべてが Apache-2.0 互換、
  (b) Windows でユーザーが `pip install atlasmith[ml]` 相当で導入できる、
  (c) AI生成キャラクターで幾何手法より明確に良い分割が得られる。

## 候補

| 候補 | 概要 | 主な根拠 | 主な懸念 |
|---|---|---|---|
| **SAM2 多視点自前実装** | 多視点レンダリング→SAM2で2Dマスク→メッシュ面へ逆投影→視点間融合 | Apache-2.0(コード・重み)。**wheelのみで導入可**。SAMPart3D も内部で同じ発想を使う | 実装数百行。遮蔽/凹部が見えない・視点間不整合。品質未検証 |
| **SAMPart3D**(Pointcept) | 学習済み backbone + per-object MLP 蒸留 | コード MIT・重み MIT・backbone も MIT/Objaverse 学習。SOTA・スケールで粒度制御 | **導入が重い**(下記)。面ラベル出力の公式経路なし。Windows 公式未対応 |
| **SATR** | zero-shot(専用重みを持たない) | MIT。重みが無いのでデータセット汚染が原理的に起きない | PyTorch3D 依存(Windows 公式非対応)。テキストクエリ必要 |
| **Find3D** | 点群部位分割 | コード/重み MIT、Objaverse 自動注釈で PartNet 非依存 | 点群専用・CUDA必須(CPU不可と明記)・テキストクエリ必要 |
| **何もしない(幾何のみ)** | Phase 2 計画v2 のまま | 追加依存ゼロ・実装済み設計 | **ユーザーの主用途(有機キャラ)で機能しない** |

## 証拠等級付き比較

### ライセンス判定(**コード・重み・学習データを別々に**確認 — ここが本調査の核心)

| 候補 | コード | 重み | 学習データ | 判定 |
|---|---|---|---|---|
| SAM2 | Apache-2.0【外部】LICENSE実確認 | Apache-2.0【外部】 | — | **互換 YES** |
| SAM3 | 独自 "SAM License"(軍事用途禁止等)【外部】LICENSE実確認 | 同左 | — | **NO**(SAM2と混同注意) |
| SAMPart3D | MIT【外部】 | MIT【外部】HFタグ | Objaverse への DINOv2 蒸留。PartNet/ShapeNet **不使用**【外部】論文本文 | **互換 YES** |
| tiny-cuda-nn(SAMPart3D必須依存) | BSD-3-Clause【実測】LICENSE取得 | — | — | **互換 YES** |
| SATR | MIT【外部】 | 重み無し(zero-shot) | — | **互換 YES** |
| Find3D | MIT【外部】 | MIT【外部】 | Objaverse 自動注釈【外部】論文本文 | **互換 YES** |
| **Point-SAM** | MIT【外部】 | — | **PartNet/PartNet-Mobility/ShapeNet/ScanNet(全て非商用研究限定)**【外部】論文本文 | **NO** |
| **PartSLIP/++** | MIT【外部】 | — | PartNet-Ensembled(非商用)【外部】 | **NO** |
| **P3-SAM/Hunyuan3D-Part** | Tencent独自(商用MAU上限・EU/UK/韓国除外)【外部】LICENSE実確認 | 同左 | — | **NO** |
| **PartField**(NVIDIA) | NVIDIA License §3.3 **非商用限定**【外部】LICENSE実文言 | 明示なし | — | **NO** |
| **PartSAM** | MITバッジ【外部】 | MIT【外部】 | 不明 | **非推奨** — PartField のコードをベースにしたと README で自認【外部】。ライセンス階層化リスク |
| **PartDistill** | **LICENSEファイル不在**【実測】404確認 | — | — | **NO**(無許諾) |
| **PointNet++/PointTransformer 系 part-seg 全般** | 実装ごと | — | ShapeNetPart/PartNet【外部】 | **NO**(一律) |
| HoloPart | MIT【外部】 | MIT【外部】 | — | **タスク不一致** — part *completion* であり分割をしない【外部】 |

**この表の最大の教訓**: 「コードは MIT」に釣られてはいけない。**候補の過半が、重みの学習データ
(PartNet/ShapeNet/ScanNet)の非商用条項で脱落した**。ShapeNet の Terms of Use は「非営利の研究・
教育目的に限る」かつ「営利企業に雇用されている研究者はその雇用主も拘束される」【外部】。
生き残ったのは **PartNet/ShapeNet で学習していないもの**(Objaverse 系・zero-shot)だけ。

### SAMPart3D 導入の実態【実測】(spike 2026-07-26、リポジトリを clone して実ファイル確認)

| 項目 | 実測結果 |
|---|---|
| RAPIDS(cudf/cuml) | **Windows wheel ゼロ**(105個すべて manylinux)。ただし **HDBSCAN にしか使われておらず**、作者が `# from sklearn.cluster import HDBSCAN` をコメントで残している(`pointcept/engines/{train,eval}.py`)→ **2行差し替えで回避可能** |
| flash-attention | `try/except ImportError` で既に optional 化済み(`PTv3Object.py`)→ 無くても動く |
| tinycudann | `SAMPart3D.py:14` で**直 import(必須)**。ライセンスは BSD-3 で問題なし。ただし CUDA ソースビルドが必要 |
| pointops | `python setup.py install` の CUDA 拡張ビルドが必要 |
| **CUDA Toolkit** | 導入済み nvcc 11.3 は **compute_86 まで**【実測 `nvcc --list-gpu-arch`】。RTX 4080 は **compute_89** → **このままではビルド不可**。CUDA 12.1 の導入が前提 |
| **実行方式** | **1回の推論ではない**。Blender で16視点レンダリング → **メッシュごとに MLP を学習**。**約6分/メッシュ**【外部】 |
| **面ラベル出力** | **公式経路なし**。出力は点群ラベル(.npy)。issue #26(まさにこれを問う)は**未回答**【外部】 |
| Windows サポート | issue 検索 "windows" **0件**。環境構築の issue #33・VRAM の #24 とも**メンテナ未回答**。動作確認は 24GB RTX 4090 のみで **16GB 実績不明**【外部】 |

**この spike で判断が変わった点**: 当初 SAMPart3D を「実装工数ほぼゼロ」と見積もったが、
**面ラベル変換を自前で書く必要があり**、その上に Blender 依存・ソースビルド4件・per-object 学習
6分が乗る。**「実装が楽」という最大の利点が消えた。**

### SAM2 経路の周辺【外部】

- ヘッドレスレンダラ: **pyrender は Windows で headless 不可**(OSMesa非対応)、**nvdiffrast は
  非商用限定** → **moderngl(MIT・Windows prebuilt wheel あり)が唯一の現実解**。
- 先行実装: SAM3D(MIT)/SAMPro3D(Apache-2.0)/Segment3D(Apache-2.0)はいずれも **ScanNet型の
  姿勢付きRGB-Dシーン向け**で、単一メッシュ+合成レンダリングとは前提が異なる。
  MeshSegmenter(ECCV2024)が用途は最も近いが **LICENSE 不在**【実測】でコード流用不可。
- **重要な構造的洞察**: SAMPart3D 自身が「多視点レンダリング → SAM でマスク → 3D融合」を
  内部で行っている。**SAM2 自前実装は別手法ではなく、その軽量版**。SAMPart3D の上乗せ分は
  学習済み backbone と per-object MLP 蒸留(=スケール制御と3D整合性)。

### optional ML依存の設計前例【外部】

- **InsightFace 型**(主流): 本体コード MIT・**重みは非商用**・アダプタコードは本体同梱・
  実行時ダウンロード。注記は model_zoo の README。
- **アンチパターン実測**: Microsoft TRELLIS.2 / TencentARC Pixal3D は MIT 表記のまま非商用依存
  (nvdiffrast・RMBG-2.0)を抱え、商用可否の issue が**メンテナ無回答で放置**。
  → 注記が不十分だと何が起きるかの実例。

## プロトタイプコスト

- **SAM2 経路の最小 spike**(推奨・未実施): メッシュを数視点レンダリング → SAM2 → 部位らしく
  分かれるか目視。**全部品が最終実装で再利用可能**、インストールは wheel のみ。
  判定基準: AI生成キャラクターで頭/胴/四肢が視認できる粒度で分離されるか。
- **SAMPart3D 検証**(未実施): WSL2+Ubuntu 導入から。Linux なら wheel が揃うため導入は素直だが
  数時間規模。**ただし検証できても Windows 配布性の問題は残る**。
- **共通の前提**: 実測には**実際のAI生成メッシュ**が必要(コテージ/岩スキャンでは難所にならない)。
  ユーザーが Tripo/Meshy で新規生成予定(2026-07-26 時点で未入手)。

## Advisor往復

なし(本記録時点)。方向決定時に非メイン側AIへ相談する余地あり。

## 決定記録

- **採用**: **SAM2 多視点自前実装**(2026-07-27 ユーザー決定)。
  決め手は**配布性**: SAMPart3D は「実装工数ほぼゼロ」という採用理由が spike で消滅し
  (面ラベル変換は結局自前・その上に Blender 依存・ソースビルド4件・per-object 学習6分)、
  一方 SAM2 経路は wheel 中心で Windows にそのまま入る。Atlasmith は他人が使うツールであり、
  `pip install atlasmith[ml]` が通らないバックエンドは実質使われない。
  品質面の劣後(遮蔽・視点間不整合)は受け入れ、実測しながら改善する。
- **却下(確定)**: PartField / P3-SAM・Hunyuan3D-Part / Point-SAM / PartSLIP系 /
  PartDistill / ShapeNetPart・PartNet 学習の point-seg 全般 — いずれもライセンス非互換。
  HoloPart はタスク不一致。SAM3 は SAM2 と別ライセンスにつき使用しない。
- **見送り(将来の追加候補)**: SAMPart3D。品質は SOTA でライセンスも互換だが、Linux/WSL2 前提。
  `SegmentationBackend` 抽象があるため、**将来「上級者向けの追加バックエンド」として併存可能**
  (この道は閉じない)。

### SAM2 経路の導入実測【実測】(2026-07-27)

| 項目 | 結果 |
|---|---|
| **moderngl ヘッドレス描画** | **成功**。`create_context(standalone=True)` がウィンドウ無しで通り、`GL_VERSION = 3.3.0 NVIDIA 591.86` / `GL_RENDERER = NVIDIA GeForce RTX 4080/PCIe/SSE2`。64x64 のオフスクリーン描画でピクセル値も期待どおり(0.2/0.4/0.6 → 51/102/153)。**Windows 配布性の柱が実証された** |
| moderngl 配布 | MIT・win_amd64 wheel が cp38〜cp313 で提供【実測 PyPI】 |
| sam2 配布 | PyPI に `sam2 1.1.0`・Apache 2.0。**sdist のみ(wheel 無し)だがコンパイラ不要でビルド通過**【実測】 |
| **sam2 導入(隔離venv・Python 3.12)** | **成功**。`uv pip install torch torchvision --index-url .../cu124` → `uv pip install sam2` の**2コマンドのみ**。torch 2.6.0+cu124 / `cuda avail=True` / RTX 4080 / **sm_89** 認識 |
| **重みの取得** | `SAM2ImagePredictor.from_pretrained('facebook/sam2-hiera-tiny')` が **HF から認証不要でDL+ロード、14.8秒**【実測】 |
| **推論の正しさ** | 256x256 の合成画像(中央 128² の矩形)に点プロンプト → **best mask のピクセル数 16384 = 128² と厳密一致**。**0.38秒/画像**(GPU)【実測】 |
| **自動マスク生成** | `SAM2AutomaticMaskGenerator` 利用可。`points_per_side=16` で **1.09秒**、90²=8100 の矩形3つを **8100/8100/8099** で検出【実測】。粒度制御パラメータ: points_per_side / pred_iou_thresh / stability_score_thresh / box_nms_thresh / crop_n_layers / min_mask_region_area |
| 処理時間の見込み | 0.38秒/画像 × 16視点 ≒ **6秒/メッシュ**。SAMPart3D の **6分/メッシュ** と桁違い |

### torch の Windows 配布事情【実測 2026-07-27】— 重要な留保

**素の PyPI から入る Windows 版 torch は CPU 専用**。実測:

```
uv pip install torch          # index-url 指定なし
version = 2.13.0+cpu
cuda available = False
compiled with CUDA = None
```

裏付け(PyPI メタデータ実測): torch の CUDA 依存はすべて `platform_system == "Linux"` で
ゲートされており(`cuda-toolkit==13.0.3; platform_system == "Linux"`、`nvidia-cudnn-cu13`、
`nvidia-nccl-cu13` 等)、**Windows 向けの CUDA 依存は1件も無い**。wheel サイズも
win_amd64 が 116MB に対し manylinux_x86_64 は 502MB。

**帰結**: `pip install atlasmith[ml]` を Windows で実行すると **CPU 版 torch が入る**。
SAM2 の自動マスク生成は1画像あたり多数のプロンプトを走らせるため、CPU では実用にならない。
**Windows ユーザーには `--index-url https://download.pytorch.org/whl/cuXXX` の明示が必要**。

**この留保は SAM2 採用の判断を覆さない**: SAMPart3D が要求したのは CUDA Toolkit の新規導入+
MSVC+ソースビルド4件だったのに対し、こちらは **index を1つ指定するだけ**。桁が違う。
ただし README/インストール手順にこの1ステップを明記することが**必須**(黙っていると
「入ったのに遅い」という最悪の体験になる)。対処方針(README 明記 / `[tool.uv.sources]` /
実行時の CPU 検出警告)は計画 v3 の承認事項として裁定する。

**結論(実測ベース)**: SAM2 経路は **Windows でコンパイラ不要・wheel 中心・認証不要**で成立する
(torch の index 指定という1ステップの留保付き)。工学的リスクはほぼ解消。
**残る唯一の未検証は「実際のAI生成キャラクターで意味のある部位に分かれるか」**(検証用メッシュ待ち)。
- **再評価トリガー**:
  - AI生成メッシュでの SAM2 経路の品質実測が出たとき(最優先)
  - SAMPart3D が面ラベル出力の公式経路を提供 or Windows 対応を表明したとき(issue #26/#33 の動向)
  - Apache-2.0 互換で**一回推論**の mesh part segmentation モデルが登場したとき
  - Objaverse 学習重みのライセンス継承について法的整理が進んだとき(現在は業界共通の未確定論点)
