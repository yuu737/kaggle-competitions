# NeuroGolf 2026 — 🥉 275 / 2,963

[大会ページ](https://www.kaggle.com/competitions/neurogolf-2026)

## 課題

ARC-AGI（図形パズルのベンチマーク）の400タスクそれぞれについて、変換を正確に再現する**できるだけ小さいニューラルネット**をONNX形式で提出する大会です。スコアは各タスクの `25 − ln(コスト)` の合計で、正しく解けることが前提、そのうえでメモリとパラメータを削るほど点が上がります。

## 工夫した解法（[`candidates/`](candidates/)）

公開ノートブックの解答を土台に、タスクごとにグラフを設計し直し、公式の判定で出力が一致するものだけを差し替えました。

| タスク | 変更 | コスト | 得点の増分 |
| --- | --- | ---: | ---: |
| [task187](candidates/task187_bitclosure/) | 13回の塗りつぶしを、ビット詰めした「横・縦・横」の区間閉包に置き換え | 36,814 → 25,121 | +0.382 |
| [task286](candidates/task286_compact_frontend/) | 色の復号・切り出し・詰め込みの前段を小さく作り直し | 47,001 → 33,009 | +0.353 |
| [task209](candidates/C5v2_task209/) | 20×20のインデックス面と1次元の平面を除去し、縮尺の判定を対角のサンプルで行う | 27,956 → 16,233 | +0.544 |
| [task029](candidates/C11_task029/) | 新しい設計を試したが、ONNX RuntimeのOneHotがINT64しか受けないため逆にコスト増。**不採用** | 5,525 → 25,840 | −1.54 |

各フォルダに、ONNXを組み立てるスクリプト、生成したONNX、検証結果（全例の正解数、公式コスト、過学習チェック）を入れています。task029のように、全検査は通っても得点が下がる候補は記録を残して捨てています。

## 検証（[`scripts/`](scripts/)）

- [`audit_onnx_bundle.py`](scripts/audit_onnx_bundle.py): 400個のONNXを、公式チェッカー・使える演算子・サイズ上限で検査
- [`validate_bundle.py`](scripts/validate_bundle.py): 学習・テスト・ARC-GENの例で実行して正解数を測る
- [`score_bundle.py`](scripts/score_bundle.py): 公式と同じ方法でコストと得点を計算
- [`check_overfit.py`](scripts/check_overfit.py): 既知の例を丸暗記する方式を検出。丸暗記はローカルでは全問正解でも非公開テストで大きく落ちる（実際に一度 7,133 点まで落ちた）

候補は [8段階の検査](agent-workflow/VALIDATION_PROTOCOL.md) をすべて通らない限り本線に入りません。

分かった限界: 入力がfloat32で与えられるため、画素ごとの集計は最低でも3,600バイトかかり、それ以上は削れない。上位の公開解答はすでにこの限界近くまで最適化されていた。

最終スコア 7,403（333回提出）。

## Claude Codeの使い方

Claude Codeを研究リードにし、Sonnetのサブエージェントと、Skill経由で呼ぶCLIのworkerに実装と検証を任せました。

- [`.claude/agents/neurogolf-worker.md`](.claude/agents/neurogolf-worker.md): 実装と検証を担当し、採否は決めないworker
- [`.claude/skills/`](.claude/skills/): 監査、統合、公開コードの取り込み、Codexへの委任、提出など8つのSkill
- [`agent-workflow/`](agent-workflow/): Codexへのタスク票の雛形、結果JSONの形式、検査の手順、既知の失敗、1セッションのまとめ（[2026-07-04](agent-workflow/SESSION_SUMMARY_2026-07-04.md): +1.19点）

詳しくは [CLAUDE_CODE.md](../CLAUDE_CODE.md)。

## 参考

公開ノートブック（ONNXのグラフ手術、ルールベースの生成、回路の構成など）を土台・参考にした。
