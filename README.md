# Atlasmith

AI生成3Dモデル(Tripo/Meshy/Hunyuan3D 等)の乱雑な UV を部位単位のアイランドへ再編成し、
既存テクスチャを新UVへ焼き直す Python 製 CLI ツール。パイプラインは5段(①部位分割 ②シーム決定
③平面展開 ④パッキング ⑤テクスチャ焼き直し)。現状は Phase 0(GLB/glTF/OBJ 入出力)+
Phase 1(⑤テクスチャ焼き直し)+ Phase 2(①部位分割・④部位単位パッキング)実装済み。
**②シーム決定・③平面展開の独立化は Phase 3 で行う**(現状は部位ごとに xatlas の展開を呼んでいる)。

## インストール

[uv](https://docs.astral.sh/uv/) 前提。Python 3.12(開発 pin)は `uv sync` 実行時に uv が
自動で導入する(手動インストール不要)。

```
uv sync
```

これだけで**幾何ベースの部位分割**(`--segmenter geometric`)まで含めて動く。追加依存は不要。

### ML 部位分割(SAM2)を使う場合 — optional

SAM2 多視点部位分割は optional 依存 `[ml]`(torch / sam2 / moderngl / huggingface_hub)を要する。
**Windows で GPU を使うには2手順**が必要:

```
uv sync --extra ml

uv pip install "torch==2.13.0" "torchvision==0.28.0" \
    --index-url https://download.pytorch.org/whl/cu126 --reinstall --no-deps
```

- **2手順目が必須。** 素の `uv pip install torch` は **Windows では CPU 版**になる
  (PyPI 上の torch は CUDA 依存がすべて `platform_system == "Linux"` でゲートされており、
  Windows 向けの CUDA 依存が1件も無いため)。CPU では SAM2 の自動マスク生成は実用にならない。
- **`--no-deps` が必須。** 付けないと PyTorch の index が PyPI ミラーを兼ねているために
  **必須依存が巻き戻る**(実測: numpy 2.5.1 → 2.4.4 / Pillow 12.3.0 → 12.2.0)。
- **`cu124` は使えない。** `uv.lock` が固定する torch 2.13.0 の cp312/win_amd64 wheel は
  **cu126 と cu130 にしか存在しない**(実測)。ドライバ要求が低い cu126 を採用している。
- 導入後の確認は**自動同期を避けて**行う(素の `uv run` は実行前に同期して CUDA 版を CPU 版へ
  巻き戻す):

  ```
  uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
  ```

- **モデルの重みは同梱していない。初回実行時に HuggingFace Hub から自動取得する**
  (既定 `facebook/sam2.1-hiera-large`、認証不要)。オフライン環境では最初の1回が失敗する。

## 使い方

```
uv sync
uv run python examples/make_demo_assets.py
uv run atlasmith examples/demo.glb -o examples/demo_repacked.glb --padding 8
```

1行目で依存を導入し、2行目でテクスチャ付きデモメッシュ `examples/demo.glb` を
ローカル生成し、3行目でそれを読み込んで UV を再展開しテクスチャを焼き直した
`examples/demo_repacked.glb` を書き出す(読込→部位分割→UV再展開→テクスチャ焼き直し→書出の
一気通貫パイプライン)。

### 主なオプション

| フラグ | 既定 | 意味 |
|---|---|---|
| `--texture-size` | 1024 | 出力テクスチャの一辺(テクセル) |
| `--padding` | 8 | チャート間パディング兼ガター膨張 |
| `--granularity {part,naive}` | `part` | `part` = 部位ごとに展開し、**各 UV アイランドが単一部位に収まる**ようにする。`naive` = メッシュ全体を一括展開する Phase 1 の挙動 |
| `--segmenter {geometric,sam2}` | **`sam2`** | 部位分割バックエンド。`geometric` = 二面角クラスタリング(追加依存なし・高速・決定的)。`sam2` = SAM2 多視点(`[ml]` が必要) |
| `--seg-angle` | バックエンド既定 | 二面角しきい値(度)。両バックエンドで有効(`sam2` では幾何プライアとして働く) |
| `--seg-min-faces` | バックエンド既定 | これより小さい部位を隣へマージする。両バックエンドで有効 |
| `--seg-views` | バックエンド既定 | レンダリングする視点数(`--segmenter sam2` 専用) |
| `--seg-model` | `facebook/sam2.1-hiera-large` | SAM2 のモデル id(`--segmenter sam2` 専用) |

**効かないフラグは黙って無視せずエラーにする**(いずれも**明示指定したときだけ**判定する):

- `--granularity naive` + `--segmenter` / `--seg-*` を明示(naive 経路は部位分割をしない)。
- `--segmenter geometric` を**明示** + `--seg-views` / `--seg-model` を明示
  (この2つは sam2 専用)。

**`[ml]` を導入していない環境**では、`--segmenter` 未指定(既定 `sam2`)のときだけ
**警告を出して `geometric` へフォールバック**する。`--segmenter sam2` を**明示**した場合は
黙って落とさず `ImportError` で停止し、解決方法を提示する。

### Python API

```python
from atlasmith import rebake

rebake("in.glb", "out.glb", texture_size=1024, padding_px=8, granularity="part")
```

**API と CLI で既定のバックエンドが異なる。** `rebake(..., segmentation=None)` は
**幾何バックエンド(`DihedralSegmenter`)** を使う(CLI の既定は `sam2`)。API から SAM2 を
使う場合は明示的に渡す。**渡した backend は `rebake` が閉じない**ので、閉じるのは呼び出し側:

```python
from atlasmith import rebake
from atlasmith.segmentation.multiview import make_sam2_segmenter

with make_sam2_segmenter() as segmenter:
    rebake("in.glb", "out.glb", segmentation=segmenter)
```

## 制約と限界(**使う前に読む前提**)

### 承認済みの正式制約

- 対応範囲は単一メッシュ・単一マテリアル・単一UVセットに限定する。
- **重複面・非多様体を含むメッシュは非対応**(面対応を一意に決められないため、黙って誤った
  焼き直しを返さずエラーで停止する)。
- normal map は転写するが警告付き(UV 変更でタンジェント空間の基底が変わるため、
  照明的正しさは保証しない)。
- OBJ 書き出しは basecolor のみ保持する(trimesh の `SimpleMaterial` の制約により
  normal・metallic_roughness は落ちる)。

### 部位分割の限界(実測)

- **幾何バックエンドは滑らかな有機的キャラクターでは部位に分けられない**(P=1 に退化する)。
  切れるのは「鋭い折れ目」と「別シェル」だけ。**この退化が SAM2 を入れた動機**である。
  逆にハードサーフェスなら幾何のほうが素直で速い。
- **SAM2 バックエンドは非決定的**(同じ入力で結果が変わりうる)。非決定なのは SAM2 の
  マスク提案の1段だけで、レンダリング・逆投影・融合・パッキングは決定的。
- **遅い。** 既定モデル `sam2.1-hiera-large` で **約4分/メッシュ**(24視点 × 2チャンネル、
  RTX 4080 実測)。`--segmenter geometric` なら数秒。
- **部分的に遮蔽される領域では過分割されうる**(視点から見えない辺がカットとして残るため)。
  緩和は `--seg-min-faces` による小部位マージのみ。
- **面の順序に依存する。** 同じ形状でもエクスポータが面順を変えると SAM2 経路の部位割りは
  変わりうる(融合の tie-break が面 index に依存するため)。幾何バックエンドにこの性質は無い。
- 出力アトラスの実寸法は `--texture-size` を**超えることがある**(xatlas の `resolution` は
  上限ではなく密度ヒント)。UV は縮小して収めるので出力は常に `--texture-size` の正方形になるが、
  **実効テクセル密度は要求より下がる**。ガターが 0 テクセルに潰れる場合だけ警告が出る。

## 開発

```
uv run ruff format --check .    # フォーマット
uv run ruff check .             # リント
uv run pytest                   # テスト
uv run pytest -m "not ml and not gl" -q   # GPU / ML 抜き(CI と同じ範囲)
```

`ml` / `gl` マーカーの付いたテストは GPU と `[ml]` extras を要する。CI は extras 無しで走る。

## ライセンス

Atlasmith 本体は **Apache-2.0**(全文は `LICENSE`)。帰属表示は `NOTICE`。

optional な `[ml]` 依存(torch / sam2 / moderngl / huggingface_hub)と SAM2 の重みの
ライセンス、Linux で同時に入る NVIDIA 独自ライセンスのパッケージについては
**[`Docs/licenses/THIRD-PARTY-ML.md`](Docs/licenses/THIRD-PARTY-ML.md)** に
一次資料・等級・留保つきで記録してある。要点:

- `[ml]` の直接依存はすべて permissive。**GPL / AGPL / LGPL は 0 件。**
- **SAM2 の重みは同梱せず、初回実行時に HuggingFace Hub から取得する**(重みも Apache-2.0)。
- Linux では torch の推移的依存として NVIDIA 独自ライセンスのパッケージが入る
  (Atlasmith は再配布しない)。
