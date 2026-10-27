# ROGII Wellbore Geology Prediction — 🥉 437 / 6,125

[大会ページ](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)

## 課題

水平に掘った井戸のガンマ線（GR）の記録から、地層の中での位置（TVT）を推定する回帰問題です。基準となる縦井戸（typewell）のGRと見比べて「今どの地層にいるか」を追跡する、地質の逆推定（ジオステアリング）にあたります。

## 解法

パーティクルフィルタ（PF）で地層位置を追跡し、スタッカーと周辺の井戸からの空間補間を組み合わせた。

- **パーティクルフィルタ**（[`src/geosteer_pf.py`](src/geosteer_pf.py)）: 地層位置と傾きを状態に持ち、各深さでGRの一致度から粒子を重み付けする。GRの形が似た地層が複数ある曖昧さを、粒子の分布のまま扱える。
- **比較した他の方法**: 動的計画法、HMM、CRF、ビームサーチ、ニューラルトラッカー（`src/geosteer_*.py`、`neural_tracker.py`）
- **空間補間**: 近くの井戸から地層面を推定して補正する（`geosteer_spatial*.py`）

## Working Note（[WORKING_NOTE.md](WORKING_NOTE.md)）

51件の実験を「中止条件を先に決める」「井戸単位で検証を分ける」方針で回し、**どこまでなら情報を取り出せるか**の境界をまとめた。

- 残った誤差の約74%は、井戸ごとの「直線のずれ」（基準の高さと傾き）だった。
- それを直すために試した11の方法のうち、8つは理由を測ったうえで失敗、2つは「今の方式を変えない」が結論、1つはローカルでは改善したが本番の順位表では確認できなかった。
- 学び: 確率的なフィルタは少ない乱数シードの評価だと結論が逆転する。ローカルで0.1未満の改善は確定扱いしない。

## 最終提出（[`notebook/`](notebook/)）

[`rogii-7159-plus-spatial.ipynb`](notebook/rogii-7159-plus-spatial.ipynb) が順位表で7.106を出した提出版です（出力は削除済み）。公開ノートブック（ravaghiさんの学習済み成果物とkoolbox）の上に、パーティクルフィルタの結果、空間補間、井戸ごとの形状の整合をかける層を加えています。

## Claude Codeの使い方

Claude Codeを研究リードにし、Sonnetのサブエージェントに実装と検証を任せた。

- [`agent-workflow/CLAUDE.md`](agent-workflow/CLAUDE.md): エージェントへの必須ルール。NeuroGolfの上位チームの解法記事から、分野に依存しない進め方を抜き出して作った（[移植メモ](agent-workflow/NEUROGOLF_TRANSFER_NOTES.md)、[PLAYBOOK](agent-workflow/PLAYBOOK.md)）
- [`.claude/agents/rogii-worker.md`](.claude/agents/rogii-worker.md): 実験の実装、井戸単位のCV、ノートブックの準備を担当し、採否と提出は決めないworker
- 実験の台帳と中止条件を状態ファイルに残し、セッションをまたいでも引き継げるようにした

詳しくは [CLAUDE_CODE.md](../CLAUDE_CODE.md)。
