# Claude Codeの使い方

Kaggleの3大会（NeuroGolf、ROGII、RSNA）で、Claude Codeをどう使ったかの記録です。

## 期間

| 大会 | 期間 | 結果 |
| --- | --- | --- |
| NeuroGolf 2026 | 2026-06-28 〜 07-16 | 🥉 275 / 2,963 |
| ROGII Wellbore | 2026-06-27 〜 07-23 | 🥉 437 / 6,125 |
| RSNA Knee | 2026-10-12 〜 10-22 | 426 / 6,425（最終結果待ち） |

## 1. 研究リードとworkerを分ける（NeuroGolf）

400タスクのONNXを1つずつ書き換える大会で、作業量が多く、1つの会話では足りませんでした。そこで役割を3段に分けました。

| 役割 | 担当 | やること | やらないこと |
| --- | --- | --- | --- |
| 自分 | — | 方針、提出、最終の採否 | — |
| 研究リード | Claude Code（メイン） | タスクの優先順位、ARCのルール解釈、候補の採否、統合 | 機械的な実装 |
| worker | Claude Codeのサブエージェント（Sonnet） | 実装、400タスクの走査、ONNXの解析、検証 | 採否の判断 |
| 外部worker | Codex CLI（Skillから呼び出し） | 独立した候補の実装と証拠集め | 採否の判断、本番の提出ファイルの変更 |

- サブエージェントの定義: [`neurogolf-2026/.claude/agents/neurogolf-worker.md`](neurogolf-2026/.claude/agents/neurogolf-worker.md)
- Codexへの委任: Skill [`neurogolf-delegate-codex`](neurogolf-2026/.claude/skills/neurogolf-delegate-codex/SKILL.md) が、[タスク票の雛形](neurogolf-2026/agent-workflow/CODEX_TASK_TEMPLATE.md)から依頼を書き、ラッパー（[`scripts/fable5_delegate_codex.py`](neurogolf-2026/scripts/fable5_delegate_codex.py)）でCodexを起動します。結果は[決まった形式のJSON](neurogolf-2026/agent-workflow/CODEX_RESULT_SCHEMA.json)で返り、基準ファイルのSHAが変わっていないか、禁止操作がないかを確認してから、リードが採否を決めます。
- 独立した候補はUUID付きのフォルダで並列に回し、提出ファイルの統合だけはリードが1か所で行います。

## 2. 手順をSkillにする（NeuroGolf）

毎回同じ検査を通すため、繰り返す手順を8つのプロジェクトSkillにしました（[`.claude/skills/`](neurogolf-2026/.claude/skills/)）。

| Skill | 内容 |
| --- | --- |
| `neurogolf-audit-bundle` | 400ファイルの構造、ONNXの規則、公式コスト、実行結果、丸暗記パターンの検出 |
| `neurogolf-task-lab` | 1タスク分の研究パケット（データの統計、ONNXのグラフ、演算子）を作る |
| `neurogolf-merge-candidate` | 検証済みの安い候補だけを本線と統合する |
| `neurogolf-public-refresh` | 公開ノートブックを探して取り込み、400タスクを比較してよい方を残す |
| `neurogolf-delegate-codex` | Codexへの委任 |
| `neurogolf-submit-kaggle` | 明示的に承認したZIPだけを提出し、SHAと結果を記録する |
| `neurogolf-publish-notebooks` | 公開ノートブックのシリーズを作る |
| `neurogolf-build-handoff` | 次のセッションへの引き継ぎ資料を作り直して検証する |

候補は[8段階の検査](neurogolf-2026/agent-workflow/VALIDATION_PROTOCOL.md)をすべて通らない限り本線に入りません。過去の失敗は[既知の失敗](neurogolf-2026/agent-workflow/KNOWN_FAILURES.md)に残し、同じ失敗を繰り返さないようにしています。

## 3. 前の大会の学びを指示書にする（ROGII）

NeuroGolfの上位チームの解法記事と議論を集め、分野に依存しない進め方だけを抜き出しました（[移植メモ](rogii-wellbore/agent-workflow/NEUROGOLF_TRANSFER_NOTES.md)）。それをROGIIの [`CLAUDE.md`](rogii-wellbore/agent-workflow/CLAUDE.md) の必須ルールにしています。

1. ローカルのスコアは仮説。順位表で確認できたものだけを土台にする。
2. 集計スコアが予想外なら、推測せず分割して原因を絞る。
3. 行き止まりは「なぜ死んだか」「いつ誰が閉じたか」まで記録する。
4. 委任するときは「Xを超えろ（実際に達成した例がある）」と具体的な目標を渡す。
5. 構造の書き換えと細かい調整を区別する。
6. 取り返しのつかない採用の前に、別のモデルに独立してレビューさせる。

ROGIIのworker定義: [`rogii-wellbore/.claude/agents/rogii-worker.md`](rogii-wellbore/.claude/agents/rogii-worker.md)

## 4. 調査から計画を立て直させる（RSNA）

「今のままでは銀メダルに残れない。計画を立て直せ」と指示し、Kaggle APIで順位表の全件、上位11本のノートブック、議論110件を集めさせました。そこから「0.958のチームの塊は同じ公開ノートブックのコピーで、その中の順位は運」と分析し、疑似ラベルで自前モデルを学習する方針に切り替えました（[計画書](rsna-knee/reports/PLAN_V2_20261019.md)）。

## うまくいかなかったこと

- 調査レポートが挙げた改善候補は、ルールを検証すると5件すべて不採用だった。以後は「ルールの検証が先、実装は後」を手順に入れた。
- 確率的な手法は、乱数シードが少ない評価だと結論が逆転した（ROGII）。評価のシード数を増やすことを検査の条件にした。
