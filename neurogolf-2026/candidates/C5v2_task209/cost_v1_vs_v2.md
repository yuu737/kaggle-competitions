# task209 cost: v1 vs v2

| 指標 | 現行 task209 (root submission.zip, oracle) | v1 (受理済み) | v2 (本パケット) |
|---|---:|---:|---:|
| cost (memory+params) | 27,956 | 17,734 | 16,233 |
| memory | 27,757 | 17,566 | 16,066 |
| parameters | 199 | 168 | 167 |
| nodes | - | ~204 (要確認) | 204 |
| gain vs 現行(ln) | - | 0.455148 | 0.543586 |
| v1 比 cost 削減 | - | - | -1,501 (-8.46%) |

- v1 SHA-256 (`task209_candidate.onnx`, 変更なし):
  `8530b082006dec63e180de2bc51019425483979e5e38e98aba87889fe900a097`
- v2 SHA-256 (`task209_v2.onnx`):
  `ce9f2ea65ae42635cde878de5168064317c218b95426bf1e476ae7bcc58b6f29`
- root `submission.zip` SHA-256 (変更なし):
  `1b509ed4c55c1391d63a7166a78129635292ef5c015de835714ee89dbdb4e4ca`

## 変更内容(v1 -> v2)

1. **20x20 index 面と 1D フラット面の除去(最大の削減源)**
   v1 は `label` を `Reshape` で `[400]` の 1D 面(`label_flat`, 400B)に落とし、
   `stamp`(鍵図形のstamp出力)と `kernel_label`(スケール判定用の3x5署名)の
   両方をこの `label_flat` に対する単一 `Gather`(`axis=0`)で取得していた。
   このとき `stamp` 側で使うフラットindex `src_idx` は `row*20+col` を
   `[1,1,20,20]` の **INT32 面(1,600B)** として実体化していた。
   v2 では、行オフセット(`src_r`)と列オフセット(`src_c`)がタスク上
   互いに独立(行方向は `row_off`/`scale`、列方向は `col_off`/`scale` のみに依存)
   であることを利用し、`label` に対して
   `Gather(label, src_r_vec, axis=2)` → `Gather(row_gathered, src_c_vec, axis=3)`
   の2段 Gather に置換した。これにより `label_flat`(400B)と `src_idx`
   (1,600B)を完全に除去し、代わりに `row_gathered`(400B, UINT8)という
   1枚の面だけが残る。`kernel_label` の3x5署名採取も同じ理由で
   `Gather(label, sr_vec, axis=2)` → `Gather(row_pick, sc_clip_vec, axis=3)`
   の2段 Gather に置換し、`sample_idx`(60B)を除去した。
   数学的にはどちらも「flat index = row*20+col による単一 Gather」と
   完全に同一の結果を返す(行専用インデックスと列専用インデックスに
   分解可能な場合の等価変形であり、クリップ順序も v1 と同一に保持)。

2. **試したが不採用: Einsum/Mul 入力の UINT8 化**
   `non4_u8` とその派生(`tmpl_rows0`/`tmpl_cols0`/`side_rows`/`side_cols`
   などの row/col summary ベクトル)を FLOAT32 から UINT8 に変更すれば
   理論上さらに 1,000B 超の削減が見込めたが、**現行 ONNX Runtime
   (1.27.0)の Einsum カーネルは float/double/int32/int64 のみ実装しており
   uint8/int8/int16 は `NOT_IMPLEMENTED` で失敗する**ことを実測で確認した。
   このためこの経路は v1 と同一の FLOAT32 のまま維持し、削減を断念した
   (詳細は `C5V2_SUMMARY.json` の `risks` を参照)。

## 検証カバレッジ

- `scripts/validate_rewrite_candidates.py`(root `submission.zip` の現行
  `task209.onnx` を oracle として使用): train+test+arc-gen 全266例、
  skipped=0、raw_equivalent=true、official_equivalent=true、
  candidate_correct=true。
- `onnx.checker.check_model(..., full_check=True)`: OK。
- 公式 sanitizer + 現行 ONNX Runtime セッション生成: OK。
- `scripts/check_overfit.py`: 大テーブル×低ノード数の記憶パターンなし。
- `scripts/audit_onnx_bundle.py`: 単体候補 zip として failures=0
  (400タスク中1件のみ収録のため missing=399 は想定どおり)。
- ORT 1.22.1 個別実行: 本パケットでは未実施(理由は
  `C5V2_SUMMARY.json` の `risks` 参照)。v1 は同一オペレータ集合で
  ORT 1.22.1 の 266例 load+run に成功済み(`C5_task209/runtime_ort122.json`)。
