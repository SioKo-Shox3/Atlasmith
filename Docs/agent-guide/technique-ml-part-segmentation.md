# 技術選定記録: ML部位分割バックエンド(①部位分割)

日付: 2026-07-26(調査)/ 2026-07-27(バックエンド決定・Phase 2実装再開承認) ／ 状態: **決定済み・Phase 2実装再開承認済み** ／
決定者: ユーザー承認「**調査結果見て判断します。SAM2 自前実装に移行しましょう**」(2026-07-27)。Phase 2実装再開の承認事項 A/A2/C'/E/F/G/H は 2026-07-27 付ユーザー裁定(本ファイル末尾「ユーザー裁定(2026-07-27)」節を参照)。

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

### 直接依存のライセンス一次資料確認【実測 2026-07-27、オーケストレーター実行】

Step 2-0(計画v4)の一次資料確認。PyPI JSON API と GitHub raw から取得(本セッションのオーケストレーターが
実行。本セッションの実装担当は再測していない)。

| パッケージ | 取得元 | 結果 |
|---|---|---|
| torch | PyPI JSON API | version=2.13.0 / `license=''` / `license_expression='Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT'`(**複合ライセンス**) |
| sam2 | PyPI JSON API | version=1.1.0 / `license='Apache 2.0'` |
| moderngl | PyPI JSON API | version=5.12.0 / `license='MIT'`。classifiers に `License :: OSI Approved :: MIT License` |

torch の LICENSE 全文(`https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/LICENSE`、全85行)を実取得。
冒頭 `From PyTorch:` / `Copyright (c) 2016- Facebook, Inc (Adam Paszke)` …、本文に
`Redistribution and use in source and binary forms, with or without` を含む **BSD 系文言**。
`GPL` の文字列ヒットは **0 件**。

**判定**: 直接依存3件とも Apache-2.0 互換・copyleft ゼロ。

**計画v4 §3 の訂正事項**: `Docs/plans/2026-07-27-phase2-plan-v4.md:585` は「torch = BSD-3-Clause」と
単独表記しているが、PyPI の `license_expression` は上記の**複合**(BSD-3-Clause ベース + 同梱コンポーネント
が Apache-2.0 / Apache-2.0 WITH LLVM-exception / BSD-2-Clause / BSL-1.0 / MIT の AND)。実体としてはいずれも
permissive で copyleft を含まないため絶対規則3には抵触しないが、**単独表記は不正確であり複合表記へ訂正する**
(この訂正は `Docs/plans/2026-07-27-phase2-plan-v4.md` §0-B に反映済み)。

**留保(等級を偽らない)**: 上記は**直接依存3件のみ**の確認であり、**推移的依存(`uv sync --extra ml` が引く
hydra-core / iopath / omegaconf / sympy / networkx 等数十パッケージ、および Linux 側の `nvidia-*` / `triton`
等)のライセンス検証は未実施**。この検証は計画v4 §0-A 条件8 により Step 2-1 の合否ゲート
(`uv tree --universal` / `uv.lock` パース併用)に委ねられている。

### ユーザー裁定(2026-07-27)— Phase 2 実装再開と ML 依存の承認

計画v4(`Docs/plans/2026-07-27-phase2-plan-v4.md`)§4.4 の承認事項7件について、2026-07-27 にユーザー裁定を
取得した。

| 記号 | 内容 | 裁定 | 備考 |
|---|---|---|---|
| A | Phase 2 実装再開の承認 | **再開する** | Step 2-1.5(SAM2品質spike)は実AI生成メッシュ未受領のため待機。計画v4 §0-A 条件11 により Step 2-1/2-2/2-6 は spike に依存せず並行して進む |
| A2 | optional `[ml]`(torch/sam2/moderngl)の追加 | **承認** | 推移的依存のライセンス検証を Step 2-1 の合否ゲート化。計画v4 §0-A 条件8 により Linux 側依存(`nvidia-*`/`triton` 等)も `uv tree --universal` / `uv.lock` パースで検査する |
| C' | `segmentation.multiview` の公開シンボル5 + CLI フラグ3の追加 | **承認** | トップレベル re-export は `SegmentationBackend` / `DihedralSegmenter` のみ(v2裁定Cの範囲を超えない) |
| E | 既定バックエンド | **`sam2`**(planner 推奨の `geometric` ではない) | 下記「E の帰結」参照 |
| F | `SegmentationBackend` 契約から「決定的」を必須要件から外す | **承認** | — |
| G | 重みの実行時DL許容+ライセンス注記の置き場所 | **承認(4箇所)** | `Docs/licenses/THIRD-PARTY-ML.md` / `NOTICE`(新設)/ README節 / 隔離モジュールdocstring |
| H | torch の CUDA 版導入方式 | **(1)+(3) の併用** | README に `--index-url https://download.pytorch.org/whl/cu124` 手順を明記 + 実行時 CPU 検出警告。(2) `[tool.uv.sources]` は不採用 |

B(既定 `granularity="part"`)/ D(アトラス寸法規約案(b))は計画v2で承認済みにつき再確認不要。

**E は planner 推奨(`geometric`)と異なる裁定**である。ユーザーは配布性より品質を優先し `sam2` を選んだ。
これに伴いオーケストレーターが提示した副作用2件を含め、以下3点の帰結が確定した:

1. **CLI 既定 `--segmenter` = `sam2`。**
2. **`[ml]` 未導入環境では、既定経由のときだけ `warnings.warn` して `geometric` へフォールバックする。**
   `--segmenter sam2` を明示指定した場合は計画v4 §2.6 どおり厳格に `ImportError`(黙ってフォールバックしない
   原則は明示指定経路で保たれる)。既定経由のフォールバックは警告を出すため「黙って」ではない。
   **WHY**: `.github/workflows/ci.yml:21` は `uv sync --locked`(extras無し)。既定が `sam2` で厳格
   `ImportError` だと CLI 既定パスを踏む既存 `tests/test_cli.py` の5件が CI で落ち、計画v4 §1「不変: CI は
   GPU無しランナーでgreenを維持する」と §0-A条件3(既存CLIテスト5件が無変更でgreen)の両方に正面衝突する。
3. **`rebake()` の API 既定は `segmentation=None` → `DihedralSegmenter()` のまま据え置き。**
   **WHY**: `rebake` が既定でSAM2バックエンドを構築すると `rebake` が `segmentation.multiview` をimport
   することになり、計画v4 §2.1の依存方向(`atlasmith(rebake) → segmentation`、numpyのみ)が壊れる。§4.1
   の「`rebake` はsam2を知らない」を維持する。**したがってEはCLIレイヤの既定としてのみ実装される**。API
   とCLIで既定が異なることを `rebake` のdocstringとREADMEに明記する(Step 2-7/2-8の作業項目)。

詳細な裁定根拠と計画v4本文の書き換え箇所は `Docs/plans/2026-07-27-phase2-plan-v4.md` §0-B を参照。

## Step 2-1.5 spike(2026-07-29)— **工学的検証のみ。Go/No-Go 判定ではない**

### この節を読む前に必ず読むこと(前提を取り違えると次の判断が壊れる)

- **入力は Poly Haven の CC0 家具モデルであり、AI生成メッシュではない。**
  実 AI 生成メッシュが未受領のため、**「パイプラインが動くか」という工学的検証だけを目的に代用した**。
- **したがって本 spike は品質判定を一切行えていない。** 「SAM2 が滑らかな有機形状を意味的な部位に
  分割できるか」(= 課題定義の成功基準(c)、本記録 19行)は**依然として未検証**である。
- **Step 2-1.5 本来の Go/No-Go 判定は未実施のまま残る。**
  **家具で動いたことは Go 判定にならない**(→ 末尾「未検証のまま残るもの」)。

> **等級**: 本節の数値・観察はすべて【実測 2026-07-29、オーケストレーター実行】であり、
> **本節の執筆担当(implementer)は spike を一度も実行していない**。すべて実測の転記である。
> spike スクリプトは**スクラッチパッドに置いてありコミットしていない**(リポジトリに存在しない)。

### spike の条件

- **入力**: Poly Haven `ArmChair_01`(CC0)。faces=5626 / verts=3758 / UV あり /
  basecolor・normal・metallic_roughness いずれも 1024²。**AI生成メッシュではない**(上記)。
- **環境**: RTX 4080 / torch 2.13.0+cu126 / sam2 1.1.0 / moderngl `GL_VERSION = 3.3.0` /
  重み `facebook/sam2-hiera-tiny`。

### 工学的検証の結果【実測 2026-07-29、オーケストレーター実行】

| 項目 | 結果 |
|---|---|
| レンダ方式 | **MRT**(RGBA8×2 + depth)。`GL_MAX_COLOR_ATTACHMENTS=8`。**2パス方式とも face_id・color が一致**することを4視点で確認 |
| GPU 上の面ID sentinel | `[0,1,2,3,254,255,256,257,65534,65536,16777213,16777214]` が厳密一致(**バイト境界を跨ぐ値**を含む) |
| `coverage ⇔ face_id >= 0` | 全画素で成立(違反 0) |
| V方向の行反転 | 検査通過 |
| `visible_ratio` | **0.9899**(24視点) |
| 2パス融合 | **全辺未観測時に `DihedralSegmenter` と `np.array_equal` で厳密一致**(P=21, E=8017)— 計画v4 §6 の「幾何プライアへの完全劣化」ゲートの予行 |
| 所要時間 | **25.1秒/メッシュ**(重み 7.65 + レンダ 0.87 + SAM2 15.30 + 割当 0.25 + 融合 0.01) |

**本記録 133行の見積もり(6秒/メッシュ)の訂正**: あれは**点プロンプト1回・16視点**の値だった。
**自動マスク生成 + 1024² の実構成では 25秒/メッシュ**である。それでも SAMPart3D の **6分/メッシュ**
とは依然として桁が違い、**採用判断(決定記録)は覆らない**。

### パラメータスイープ【実測 2026-07-29、オーケストレーター実行】

| 構成 | assigned_ratio | observed_edges | masks総数 | cut_edges | P(sam2) | 総時間 |
|---|---|---|---|---|---|---|
| perspective + texture(spike既定) | 0.721 | 3,491 | 102 | 8 | 25 | 25.1s |
| **perspective + texture_normal** | **0.980** | **6,986** | 73 | 2 | 19 | 24.1s |
| perspective + texture, fov=60 | 0.734 | 3,975 | 87 | 5 | 25 | 23.4s |
| perspective + texture, fov=60, points_per_side=32 | 0.813 | 4,922 | 147 | 15 | 26 | 62.5s |

いずれも n_views=24 / image_size=1024 / points_per_side=16(最終行のみ 32)、幾何プライアは P=21。

**最大の知見**: `shading="texture_normal"` が `assigned_ratio` を **0.721 → 0.980** に引き上げ、
観測辺を **3,491 → 6,986** と倍増させた。**追加コストはゼロ**(24.1s vs 25.1s は誤差)。
これにより **計画v4 §2.4.3 の planner 推奨(`texture_normal`)が実測で裏付けられた**
(計画v4 §7 UNCERTAIN-2 の**一部が**解消。有機形状での効果は未検証なので全部ではない)。
なお「テクスチャの明暗に埋もれていた形状境界が、法線を混ぜることで SAM2 に見えるようになった」は
**理由の解釈であって実測ではない**(実測されたのは数値であり、因果ではない)。

**効かなかったもの**:

- **fov 拡大**(0.721 → 0.734)。対象を大きく写すことより shading の方が支配的。
- **`points_per_side=32`**。SAM2 が 15.5s → 55.6s と **3.6倍**になる割に assigned_ratio は 0.81 止まりで、
  費用対効果が悪い。

### 品質面の観察 — **判定ではない(対象が家具である)**

`cut_edges` は既定で **8 / 3,491**、`texture_normal` では **2 / 6,986**。
**SAM2 は椅子を「1つの物体」としてしか見ておらず、部位に切っていない。**
P(sam2)=19 < P(geometric)=21 であり、**家具では SAM2 が幾何より粗い**。

これは**想定どおりの対照結果**である。本記録の課題定義(10〜13行)が述べるとおり、幾何が得意なのは
「鋭い折れ目」と「別シェル」であり、椅子はまさにそれに当たる。
**SAM2 の価値は、幾何が P=1 に退化する滑らかな有機形状でしか測れない。**
したがってこの観察は、**「SAM2 は使えない」とも「SAM2 は使える」とも読んではならない。**

### 計画の欠落として発覚したもの

1. **`[ml]` extras に `huggingface_hub` が欠けていた**(spike 初回実行が
   `ModuleNotFoundError: No module named 'huggingface_hub'` で失敗)。`sam2` の `build_sam2_hf` は
   `huggingface_hub` を import するが sam2 自身は依存宣言していない。
   **対処済み**: `pyproject.toml` の `[ml]` に追加 / 経緯と追加9件のライセンスは
   `Docs/licenses/THIRD-PARTY-ML.md` §9。
2. **計画v4 §2.4.1 が正射影時のカメラ距離 `d` を規定していない**(half-extent = `R·1.1` しか
   書いていない)。spike では「投影行列だけが違う比較」にするため**透視と同じ `d`** を使った。
   **production の既定は Step 2-3 で確定が必要**(本 spike は既定を決めていない)。

### 未検証のまま残るもの(この節で解消していないもの)

- **実 AI 生成メッシュでの品質判定** = Step 2-1.5 本来の Go/No-Go 判定。**最重要の未検証事項。**
  **家具での工学的成功は Go 判定にならない**(成功基準(c)は未評価のまま)。
- **有機形状での `texture_normal` の効果**。上の +0.26 は家具1体・24視点の1条件での観測。
- **正射影の既定カメラ距離 `d`**(上記2 — Step 2-3 で確定)。
- 本 spike は**単一メッシュ・単一重み(`sam2-hiera-tiny`)**の結果であり、より大きい重み
  (`small` / `base-plus` / `large`)や他形状での挙動は測っていない。
