# 第三者ライセンス記録 — optional 依存 `[ml]`

**対象**: `pyproject.toml` の `[project.optional-dependencies].ml`(`torch` / `sam2` / `moderngl`)と、
それが引き込む推移的依存の全体。

**目的**: 絶対規則3(**Apache-2.0 と非互換な依存(GPL 系等)を持ち込まない** — `CLAUDE.md`)の
**検証証跡**を残すこと。計画v4(`Docs/plans/2026-07-27-phase2-plan-v4.md`)Step 2-1 の合否ゲート
【BL-7】および §0-A 条件8(Linux 側依存も検査に含める)の成果物。

**状態**: Step 2-1 時点の記録。Step 2-8 で清書予定(重みのライセンス・学習データの扱い・
再配布の有無を追記)。

---

## 等級の凡例(主張の出どころを偽らない)

| 等級 | 意味 |
|---|---|
| 【実測 YYYY-MM-DD、オーケストレーター実行】 | 本リポジトリのオーケストレーターが当開発機で実際にコマンドを走らせて得た出力。**本文書の執筆担当(implementer)は再測していない** |
| 【外部】 | 一次資料(PyPI JSON API / GitHub raw 等)から取得した事実。URL を併記 |
| 【推測】 | **確認していない**。根拠のある推定であって実測ではない |

> 本文書の数値はすべてオーケストレーターの実測を転記したものである。執筆担当は
> `uv` を一切実行していない(CUDA 版 torch の導入が並行実行中だったため意図的に回避した)。

---

## 1. 直接依存3件

【実測 2026-07-27、オーケストレーター実行 — PyPI JSON API + GitHub raw】

| パッケージ | 解決バージョン | ライセンス | 取得元 |
|---|---|---|---|
| `torch` | 2.13.0 | `license_expression` = `Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT`(**複合**) | PyPI JSON API |
| `sam2` | 1.1.0 | `license` = `Apache 2.0` | PyPI JSON API |
| `moderngl` | 5.12.0 | `license` = `MIT`、classifier に `License :: OSI Approved :: MIT License` | PyPI JSON API |

**torch の LICENSE 全文を実取得**【外部】: `https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/LICENSE`
(全85行)。本文に BSD 系の再配布条項文言を確認。文字列 `GPL` のヒットは **0 件**。

**注意(計画v4 §3 の訂正)**: 計画v4 本文は torch を「BSD-3-Clause」と**単独表記**しているが、
実体は上記の**複合**である。構成要素はいずれも permissive で copyleft を含まないため絶対規則3には
抵触しないが、単独表記は不正確。訂正は `Docs/agent-guide/technique-ml-part-segmentation.md`
「直接依存のライセンス一次資料確認」節および計画v4 §0-B に記録済み。

**判定**: 直接依存3件はすべて Apache-2.0 互換・copyleft ゼロ。

---

## 2. 当開発機(Windows)にインストールされた集合

【実測 2026-07-28、オーケストレーター実行】

`uv sync --locked --extra ml` の後、`importlib.metadata.distributions()` を走査した結果:

```
total distributions = 34
GPL/AGPL/LGPL/NonCommercial/Proprietary hits = 0
license metadata absent = 0
```

検出に使った正規表現(大文字小文字無視):

```
\b(A?GPL|LGPL|GPLv|GNU General Public|GNU Lesser|NonCommercial|non-commercial|CC BY-NC|Proprietary)\b
```

走査対象は各 dist の `License` / `License-Expression` / `Classifier: License ::` メタデータ。
**ライセンス記載が欠落している dist は 0 件**(= 34件すべてが何らかのライセンス表明を持つ)。

---

## 3.【重要】Linux 側を含む universal な検査

### 3.1 構造的な注意 — Windows の集合だけを見ると19件が検査から漏れる

> **`importlib.metadata.distributions()` は「その機械に実際に入ったもの」しか列挙しない。**
> Atlasmith の開発機は Windows であり、`torch` の CUDA 依存(`nvidia-*` / `triton` 等)は
> すべて **Linux 環境マーカー付き**なので当機には**インストールされない**。
> したがって §2 の「34件・GPL 系 0 件」だけを根拠にすると、**Linux でのみ入る19件は
> 絶対規則3 の検査を一度も通らないまま通過する**。
> これが計画v4 §0-A 条件8 が要求した検査そのものであり、実際に **NVIDIA 独自ライセンスの
> パッケージがこの19件の中から見つかった**(§3.3)。
> **今後 `uv.lock` を再解決したときも、当機のインストール集合ではなく `uv.lock` 全体を見ること。**

### 3.2 差分の実測

【実測 2026-07-28、オーケストレーター実行】`uv.lock` をパースし、当機のインストール集合と差分:

```
uv.lock total = 53
installed on Windows = 34
NOT installed here (Linux/other-platform only) = 19
```

### 3.3 Linux 専用の19件のライセンス

【外部、PyPI JSON API で個別確認 — 2026-07-28、オーケストレーター実行】

| パッケージ | ライセンス | 等級 |
|---|---|---|
| `nvidia-cusparselt-cu13` | **NVIDIA Proprietary Software** | 【外部】PyPI メタデータに明記 |
| `cuda-bindings` | **LicenseRef-NVIDIA-SOFTWARE-LICENSE** | 【外部】PyPI メタデータに明記 |
| `cuda-pathfinder` | Apache-2.0 | 【外部】PyPI メタデータに明記 |
| `triton` | MIT | 【外部】classifier `License :: OSI Approved :: MIT License` |

**残り15件は PyPI にライセンス記載が無い**:

`cuda-toolkit` / `nvidia-cublas` / `nvidia-cuda-cupti` / `nvidia-cuda-nvrtc` /
`nvidia-cuda-runtime` / `nvidia-cudnn-cu13` / `nvidia-cufft` / `nvidia-cufile` /
`nvidia-curand` / `nvidia-cusolver` / `nvidia-cusparse` / `nvidia-nccl-cu13` /
`nvidia-nvjitlink` / `nvidia-nvshmem-cu13` / `nvidia-nvtx`

> **この15件のライセンスは【推測】である。**
> PyPI のメタデータにライセンス欄が無く、これらは Windows である当開発機には
> インストールされないため、**wheel 同梱の LICENSE ファイルを実見できていない**。
> 上の2件(`nvidia-cusparselt-cu13` / `cuda-bindings`)と同系列の NVIDIA 配布物であることから
> **NVIDIA 独自ライセンスであろうと推測する**が、確認はしていない。
> **確認するには Linux 環境で `uv sync --extra ml` を行い、各 dist-info の
> `LICENSE` / `METADATA` を実見する必要がある。**

---

## 4. 判定とユーザー裁定

### 4.1 判定

- **GPL / AGPL / LGPL / 非商用限定ライセンスは、Windows 集合(34件)・Linux 専用集合(19件)の
  いずれからも検出されなかった(0件)。**
- 計画 Step 2-1 の停止条件(「GPL 系が1件でも検出されたら即停止」)には**該当しない**。
- ただし Linux 専用集合には **NVIDIA 独自(proprietary)ライセンス**が含まれる。

### 4.2 ユーザー裁定(2026-07-28)

NVIDIA 独自ライセンスの扱いについて、ユーザーから **「記録して進む」** の裁定を得た。

裁定の根拠として提示した分析(4点):

1. **GPL / AGPL / LGPL はゼロ。** 計画 Step 2-1 の停止条件には該当しない。
2. **NVIDIA ライセンスはコピーレフトではない。** Atlasmith 自身のコードに
   ライセンスが継承されることはない(Apache-2.0 のまま)。
3. **Atlasmith はこれらを同梱・再配布しない。** `pip install atlasmith[ml]` / `uv sync --extra ml`
   の際に、パッケージマネージャが PyPI から利用者の環境へ直接取得する。
4. ただし事実として、**Linux で `atlasmith[ml]` を入れると NVIDIA 独自ライセンスのバイナリが
   同時に入る**。これは利用者が知るべき情報であり、§5 として明記する。

---

## 5. 利用者への注記

**Linux 環境で `atlasmith[ml]`(optional 依存)を導入すると、NVIDIA 独自ライセンスの
バイナリパッケージが同時にインストールされます。**

- 対象は `torch` の CUDA ランタイム依存(`nvidia-*` / `cuda-bindings` / `cuda-toolkit` 等)。
- これらは Atlasmith が同梱・再配布するものではなく、パッケージマネージャが PyPI から
  取得するものです。ライセンス条件は各パッケージの配布物に従います。
- Atlasmith 本体および必須依存は Apache-2.0 互換のみで構成されており、`[ml]` を導入しなければ
  これらは一切入りません(全ての決定的経路・CI は `[ml]` 無しで動作します)。
- Windows 環境では、PyPI の既定 wheel は CPU 専用のため上記 NVIDIA パッケージは入りません
  (CUDA 版の導入手順は §6)。

---

## 6. CUDA 版 torch の導入手順(計画v4 承認事項 H の**訂正済み**版)

計画v4 承認事項 H は README に `--index-url https://download.pytorch.org/whl/cu124` を書くとしていたが、
**2点の誤りが実測で判明した**【実測 2026-07-28、オーケストレーター実行】。以下が正しい手順。

### 6.1 手順(3ステップ)

```
# 1. まず通常どおり同期する(CPU 版 torch と全依存が入る)
uv sync --extra ml

# 2. torch/torchvision だけを CUDA 版へ差し替える(--no-deps 必須)
uv pip install "torch==2.13.0" "torchvision==0.28.0" \
    --index-url https://download.pytorch.org/whl/cu126 --reinstall --no-deps

# 3. 検証は自動同期を回避して行う
uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
#   もしくは .venv\Scripts\python.exe を直接使う
```

**ステップ3の注意**: 素の `uv run` は実行前に自動同期を行い、**CUDA 版を CPU 版へ巻き戻す**。
必ず `--no-sync` を付けるか、venv の Python を直接起動すること。

### 6.2 訂正1 — `cu124` は使えない

PyTorch の各 index を実取得し、`cp312` / `win_amd64` の torch wheel を列挙した結果:

```
cu124    利用可能な torch: (該当なし)
cu126    利用可能な torch: 2.10.0, 2.11.0, 2.12.0, 2.12.1, 2.13.0
cu128    利用可能な torch: 2.10.0, 2.11.0
cu130    利用可能な torch: 2.10.0, 2.11.0, 2.12.0, 2.12.1, 2.13.0
```

`uv.lock` が固定する torch は **2.13.0** なので、選択肢は **cu126 または cu130 のみ**。
ドライバ要求が低く枯れている **cu126 を採用**する(開発機の RTX 4080 = sm_89 はどちらも対応)。

### 6.3 訂正2 — `--no-deps` が必須

PyTorch の index は PyPI のミラーも兼ねているため、`--no-deps` 無しで実行すると
**必須依存がダウングレードされる**(dry-run の実測):

```
- numpy==2.5.1              + numpy==2.4.4
- pillow==12.3.0            + pillow==12.2.0
- setuptools==83.0.0        + setuptools==78.1.0
- typing-extensions==4.16.0 + typing-extensions==4.15.0
```

`--no-deps` を付けた場合は torch/torchvision だけが置換される:

```
- torch==2.13.0        + torch==2.13.0+cu126
- torchvision==0.28.0  + torchvision==0.28.0+cu126
```

### 6.4 残課題(本文書の許可パス外のため未修正)

以下は依然として旧情報(`cu124`)を指している。**本文書 §6 が正**である。

- `pyproject.toml:17`(`[ml]` extras 直上のコメント)
- `Docs/agent-guide/technique-ml-part-segmentation.md:212`(承認事項 H の転記表)

(`README.md` には現時点で `cu124` の記述は無い — README のライセンス節・導入手順は Step 2-8 で新設予定。)

---

## 7. 再検証のトリガー

以下のいずれかが起きたら、**本文書の §2 / §3 を再実行して更新する**:

1. **`uv.lock` を再解決したとき**(`uv lock` / `uv lock --upgrade` / 依存の追加・削除・pin 変更)。
   → Windows のインストール集合だけでなく、**必ず `uv.lock` 全体をパースして Linux 側も見る**(§3.1)。
2. **torch のメジャー/マイナー更新**。CUDA ランタイム依存の構成(`nvidia-*` の顔ぶれ)が変わる。
   同時に §6.2 の index 可用性(どの `cuXXX` にどの torch があるか)も再取得する。
3. **新しい optional 依存を追加するとき**(extras の追加、`[ml]` への新パッケージ追加)。
   絶対規則1(技術選定記録 + ユーザー承認)と併せて実施する。
4. **Linux 環境が利用可能になったとき**。§3.3 の15件を【推測】から【実測】へ格上げする
   (各 dist-info の `LICENSE` / `METADATA` を実見)。
5. **配布形態が変わったとき**(Atlasmith が依存を同梱・再配布する形になった場合)。
   §4.2 の根拠3が崩れるため、裁定そのものをユーザーに再確認する。

---

## 8. 未確認事項(留保 — 検証済みのように書かない)

- **§3.3 の15件の NVIDIA パッケージのライセンスは【推測】。** 同梱 LICENSE を実見していない。
- **重み(SAM2 checkpoint)のライセンスは本文書では未検証。**
  既存記録 `Docs/agent-guide/technique-ml-part-segmentation.md`(ライセンス判定表)は
  「SAM2 のコード・重みとも Apache-2.0【外部】」としているが、**本文書ではこれを再確認していない**。
  重み・学習データ・再配布の有無の記載は Step 2-8 の清書で行う。
- **本検査は当開発機で解決された `uv.lock` の内容に対するもの。** 利用者の環境で
  異なるバージョンが解決された場合の結果は保証しない。
