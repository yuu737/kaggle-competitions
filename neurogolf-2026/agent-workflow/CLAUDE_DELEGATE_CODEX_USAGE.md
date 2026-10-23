# Claude Code側：Codex委任Skillの使い方

## 目的

`/neurogolf-delegate-codex`は、Claude Code上のFable5が限定的な作業を外部
Codex CLI workerへ委任するためのプロジェクトSkillです。

Skill本体：

```text
.claude/skills/neurogolf-delegate-codex/SKILL.md
```

## 初回準備

1. NeuroGolfルートからClaude Codeを起動します。
2. 新しくSkillを追加した直後はClaude Codeを一度再起動します。
3. Codex認証を確認します。

```powershell
.\scripts\fable5_delegate_codex.cmd --check-auth
```

認証されていない場合：

```powershell
& "$env:APPDATA\npm\codex.cmd" login --device-auth
```

## Fable5での呼び出し

Claude CodeへFable5の初期指示を渡した後、次を実行します。

```text
/neurogolf-delegate-codex
```

具体的な依頼も同時に書けます。

```text
/neurogolf-delegate-codex task118の現行ONNXを解析し、cost削減候補を実装・検証して。採用判断はせず証拠だけ返して。
```

## Fable5が行う手順

1. `FABLE5/CODEX_TASK_TEMPLATE.md`から限定的なタスクを作成する。
2. `FABLE5_WORK/codex-inbox/<task-name>.md`へ保存する。
3. wrapperを実行する。

```powershell
.\scripts\fable5_delegate_codex.cmd `
  --task-file ".\FABLE5_WORK\codex-inbox\<task-name>.md" `
  --title "<task-name>" `
  --scope experiment
```

4. 出力された`result.json`と`manifest.json`を読む。
5. baseline SHA、`guard_errors`、検証証拠を確認する。
6. Fable5自身が候補の採用・却下と次の研究方針を決める。

## 委任に向く作業

- ONNX/Python builderの実装
- 400 taskの機械的scan
- graph、operator、initializer、tensor memory解析
- candidateのcorrectness・cost・ORT検証
- ログ解析と再現可能な不具合切り分け
- JSON、CSV、manifest、比較表の作成

## Fable5側に残す判断

- ARC規則の最終解釈
- task優先順位
- 一般化の最終判断
- candidate採用・累積統合
- 7250へ向けたportfolio配分
- ユーザーへの最終報告

## 並列委任

相互に独立したtaskは、それぞれ異なるtaskファイルとtitleで実行できます。
各runにはUUID付きディレクトリが作られるため、成果物は衝突しません。

同一の`submission.zip`を変更する作業は並列化しません。通常は
`scope=experiment`で個別candidateを作り、統合はFable5が後から行います。

## 外部操作

Codex workerには以下を委任しません。

- Kaggle提出
- 公開Notebookのpublish
- credential参照
- confirmed baseline変更

`--allow-main-update`は、ユーザーが対象更新を明示的に許可した場合だけ使用します。
