# Kaggle Competitions

> Solutions and agent workflows from my Kaggle competitions (Competitions Expert / Notebooks Expert, started June 2026, 3 bronze medals). In NeuroGolf, ROGII and RSNA I ran Claude Code as a research lead that delegated implementation to Sonnet subagents and external CLI workers, with project skills and fixed validation gates. I set the direction, the acceptance rules and the final calls.

Kaggleの大会に取り組んだ記録です。2026年6月に始め、約2か月で **Competitions Expert** になりました（現在は **Notebooks Expert** も）。

- Kaggle: [yuu111111111](https://www.kaggle.com/yuu111111111)
- Competitions: 銅メダル3（3,204位 / 221,834人）
- Notebooks: 銀メダル1・銅メダル4（1,261位 / 59,178人）

## 成績

| 大会 | 種類 | 順位 | 上位 | 結果 |
| --- | --- | --- | --- | --- |
| [ROGII Wellbore Geology Prediction](rogii-wellbore/) | 時系列・地質の逆推定 | 437 / 6,125 | 7.1% | 🥉 銅 |
| [NeuroGolf 2026](neurogolf-2026/) | 最小のニューラルネット設計（ONNX） | 275 / 2,963 | 9.3% | 🥉 銅 |
| [Kaggriculture](kaggriculture/) | 対戦型エージェント | 991 / 10,148 | 9.8% | 🥉 銅 |
| [RSNA Knee Abnormality Detection](rsna-knee/) | 医用画像（膝MRI） | 402 / 6,300 | 6.4% | メダル確定待ち |
| Pokémon TCG AI Battle | 対戦型エージェント | 802 / 6,807 | 11.8% | （メダル対象外） |
| [Orbit Wars](orbit-wars/) | 対戦型エージェント | 748 / 4,730 | 15.8% | — |

順位はKaggleの表示（2026-10-27時点）。RSNAはメダルの確定待ちです。

## 公開ノートブック（NeuroGolf、銀1・銅4）

NeuroGolfの研究で分かったことを、5本のノートブックとして公開しました。どれも同じ検証済みの提出ファイル（公開最高スコアに、検証した1タスクの改良を加えたもの）を出力し、研究の別々の側面を解説しています。

| ノートブック | 内容 | 票 |
| --- | --- | ---: |
| [Anatomy of a 45-Point Climb](https://www.kaggle.com/code/yuu111111111/neurogolf-7225-19-anatomy-of-a-45-point-climb) | 約45点の伸びを、汎用の整理・公開解の統合・タスク別の再設計に分解した記録 | 69 |
| [7 Findings From the Full Research Run](https://www.kaggle.com/code/yuu111111111/neurogolf-7-findings-from-the-full-research-run) | 研究全体から得た7つの定量的な知見 | 56 |
| [6 Failure Modes That Saved the Most Time](https://www.kaggle.com/code/yuu111111111/neurogolf-6-failure-modes-that-saved-the-time) | 時間を一番節約した6つの失敗パターン | 43 |
| [Graph Polish Is Over. Redesign the Algorithm](https://www.kaggle.com/code/yuu111111111/neurogolf-graph-polish-is-over-redesign-the-alg) | Claude（Fable5）を研究リードにした実験から、細かい調整より再設計が効くと分かった話 | 25 |
| [The Experiments That Failed—and Why](https://www.kaggle.com/code/yuu111111111/neurogolf-the-experiments-that-failed-and-why) | 効かなかった最適化と、その理由の報告 | 20 |

票数は2026-10-26時点。

## Claude Codeで研究チームを組む

NeuroGolf・ROGII・RSNAでは、Claude Codeを「研究リード」にして、実装と検証を下のエージェントに任せる体制を組みました。詳しくは [CLAUDE_CODE.md](CLAUDE_CODE.md)。

```mermaid
flowchart TB
  U[自分<br/>方針・採用の基準・最終判断・提出] --> L[Claude Code 研究リード<br/>優先順位・仮説・採否]
  L -->|サブエージェント| S[Sonnet worker<br/>実装・400タスクの走査・検証]
  L -->|Skill経由で委任| C[CLI worker<br/>独立した候補の実装と証拠集め]
  S --> E[結果JSON・検証ログ・SHA]
  C --> E
  E --> L
```

- **役割の分離**: リードは「何を試すか」「採用するか」だけを決め、workerは決めない。workerの結果は判断ではなく証拠として受け取る（[neurogolf-worker](neurogolf-2026/.claude/agents/neurogolf-worker.md)）。
- **手順のSkill化**: 監査・候補の統合・公開コードの取り込み・提出などを8つのプロジェクトSkillにし、毎回同じ検査を通す（[`.claude/skills/`](neurogolf-2026/.claude/skills/)）。
- **外部エージェントへの委任**: SkillからCLIのコーディングエージェントを並列に起動し、決まった形式のJSONで結果を返させる。基準ファイルのSHAが変わっていないか、禁止操作がないかを自動で確認してから採否を決める（[neurogolf-delegate-codex](neurogolf-2026/.claude/skills/neurogolf-delegate-codex/SKILL.md)）。
- **大会をまたいだ学習**: NeuroGolfの上位チームの解法記事から、分野に依存しない進め方を抜き出し、次のROGIIの `CLAUDE.md` のルールにした（[CLAUDE.md](rogii-wellbore/agent-workflow/CLAUDE.md)、[移植メモ](rogii-wellbore/agent-workflow/NEUROGOLF_TRANSFER_NOTES.md)）。

## 自分が決めたこと

エージェントは手を動かしますが、方向を間違えることもあります。次の判断は自分が出しました。

- **丸暗記の禁止**（NeuroGolf）: 既知の例を丸暗記するモデルは、ローカルでは全問正解でも非公開テストで7,133点まで落ちた。以後、丸暗記パターンの検出を全提出の必須検査にした。
- **伸びない方向を打ち切る**（NeuroGolf）: 汎用のグラフ書き換えとカーネル縮小を実測し、合わせて+0.06点しか出ないと分かった時点で、その系統を全廃してアルゴリズムの再設計だけに絞らせた。
- **探す役と作る役を分ける**（NeuroGolf）: リードには改善点を探すことだけをさせ、実装は別のエージェントに任せる体制にした。
- **速く回せる検査から作る**（ROGII）: 数時間かかる評価ではなく、数分で判定できる小さな検査を先に設計させた。
- **予測の癖が違うモデルを組み合わせる**（RSNA）: 1本のモデルを強くするより、予測の癖が違うモデルを組み合わせるほうが効くと判断した。自作モデルは単体では弱かったが、35%混ぜた1本を最終提出に入れ、最上位の提出と同じスコアを得た。

## 共通の進め方

- **検証を先に作る**: 提出前に、公式の評価と同じ検査をローカルで再現する（NeuroGolfでは400タスク分のONNXに8段階の検査を通す: [VALIDATION_PROTOCOL.md](neurogolf-2026/agent-workflow/VALIDATION_PROTOCOL.md)）。
- **採用の基準を先に決める**: 実験ごとに「この条件なら捨てる」を決めてから回す。ROGIIでは51件の実験すべてに中止条件を置いた。
- **失敗も記録する**: 効かなかった方法と理由を残す。ROGIIの[Working Note](rogii-wellbore/WORKING_NOTE.md)は、11の方法のうち何が、どの理由で効かなかったかの記録です。

## このリポジトリについて

各大会の解法の説明、コード、エージェントへの指示書を載せています。大会データ・学習済みモデル・提出ファイル・認証情報は含めていません。公開ノートブックや公開コードを土台にした部分は、各フォルダに出典を書いています。
