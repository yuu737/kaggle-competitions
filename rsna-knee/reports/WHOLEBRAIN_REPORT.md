# RSNA Knee × MaleCNS 全脳回路 初期検証

2026年10月8日（日本時間）。全脳回路の実計算、公式58検査での通常分類器との比較、Internet無効のKaggle推論まで完了。**MRI性能の改善は確認できなかった。** 高精度の大会モデルが完成したという結果ではない。

## 主要結果

同一の固定ImageNet ResNet50特徴、公式12ラベル、検査単位の3-fold outer CVを共有。各foldの学習側だけで2-fold inner CVを行い、正則化Cを選択した。58検査を評価し、レポート、欠損ラベルの陰性化、大会を学習済みの公開重みは使用していない。

|方式|OOF macro ROC-AUC|95% bootstrap区間|fold別macroの平均|
|---|---:|---:|---:|
|通常の画像特徴＋logistic分類器|0.6053|0.5329–0.6734|0.6429|
|MaleCNS全脳＋学習readout|0.6196|0.5580–0.6785|0.6543|
|同じ入力・出力写像、回路を迂回|0.6268|0.5637–0.6854|0.6585|
|接続先をシャッフル|0.6172|0.5540–0.6775|0.6506|
|接続をゼロ化|0.6226|0.5609–0.6828|0.6581|

全脳−通常の差は+0.0143、paired 95%区間は**−0.0178～+0.0472**。全脳−迂回は−0.0072、区間−0.0217～+0.0078。全脳−シャッフルも+0.0024、区間−0.0061～+0.0108。実際の接続構造がMRI精度に寄与したという証拠は得られていない。

![比較結果](outputs/wholebrain/comparison.png)

bootstrapは検査単位・2000回・同一再標本化による方式間比較。固定OOF予測の再標本化なので、再学習・異なるseed・分割変更の不確実性は含まない。fold平均とOOFの差には、少数検査で学習したfold間の確率水準の差も影響する。評価検査は20/19/19件、MCL陽性は各2/3/4件のみ。患者IDがないため検査単位での非重複を保証するが、別検査として現れる同一患者の照合はできない。ラベルごとの結果は[per_label_auc.csv](outputs/wholebrain/per_label_auc.csv)。Medial Meniscusは全脳でも0.4483、MCLは0.5011であり、実用的な高精度は得ていない。

## 全脳は実際に計算したか

**成功。** [MaleCNS v1.0](https://male-cns.janelia.org/download/)の全165,122 Traced神経を保持し、未確定の非グリア注釈も含めて199,713状態、26,013,498有向接続、125,334,312シナプス相当の重みを使用した。中心脳・両側視葉に加えてVNCを含む。残り34,591状態を確認済み神経とは呼ばない。

入力はMRIの3平面から各最大16断面→ImageNet ResNet50→12,288特徴→各状態への4成分の固定符号付き投影。全CSR行列を用いて4回メッセージを伝播し、全最終状態を2048区画に集約して12個のsigmoid readoutを学習した。入力特徴から分類器への直接経路はない。全脳の内部接続と入力投影は固定で、学習した接続はreadoutのみ。これは生理学的に検証済みのスパイクシミュレーションではない。

- 全Traced神経の保持、元IDの対応、採用接続の件数・シナプス数の一致を検証した。
- 全199,713状態で活動が生じ、全状態から学習済み出力への非ゼロの依存を確認した。
- 実MRIで接続をゼロにすると、学習済みheadを固定した出力確率が最大0.1313変化した。シャッフルでは最大0.1380変化した。
- 最終全脳活動を消すと最大0.8738変化した。回路の計算結果を無視して予測している構成ではない。
- CPU/CUDAの全脳readout一致誤差は最大2.38×10⁻⁷。32行の手計算による接続方向・重みの照合も通った。
- 正負教師によるSGD更新と、実MRIでのreadout最適化を確認した。全脳readoutの訓練BCEは0.6931から約0.17へ低下したが、これは訓練損失であり汎化性能の証拠ではない。

元表の注釈外断片・グリアに接続する125,843,186対は対象外。採用ノード間はweight≥1を全て計算し、接続強度による刈り込みをしない。Tracedの499状態は採用グラフ内に接続がないが、入力・活動・出力を保持した。この境界、正規化、NT符号、失われる情報と数式は[データ・計算仕様](docs/wholebrain_data_design.md)に記録した。

## 資源の実測

|処理|環境|時間|ピークメモリ|
|---|---|---:|---:|
|MaleCNS全表からCSR作成|PC CPU|45.80秒|RAM 9.92GiB|
|画像前処理61検査＋表取得・変換等|Kaggle CPU|186.10秒|RAM 10.35GiB|
|画像前処理だけ61検査|Kaggle CPU|107.50秒、平均1.755秒/検査|上行に含む|
|画像エンコーダ61検査|RTX A1000|13.65秒（読込含む）、計算平均0.128秒/検査|PyTorch VRAM約195MiB|
|画像エンコーダ61検査|Kaggle T4|14.12秒（モデル読込含む）|PyTorch VRAM約194MiB、RSS約1.23GiB|
|実全脳回路61検査|RTX A1000|1.156秒、平均0.0190秒/検査|比較プロセスでVRAM約341MiB|
|5方式の特徴生成・nested CV・bootstrap等|PC|18.30秒|RAM 1.69GiB|
|エンコーダ＋全脳を同時常駐|RTX A1000、公開例キャッシュ3検査|起動含む6.39秒|RAM 1.85GiB、PyTorch VRAM509MiB、予約588MiB|
|上記同時常駐時のGPUボード使用量|nvidia-smi 24回標本|同上|標本最大681MiB|
|DICOM→12確率→submission.csv、公開例3検査|Kaggle CPU、Internet無効|42.49秒|RAM 1.44GiB、GPUなし|

ローカルGPUのメモリはPyTorch allocatorとGPUボード全体の標本値を区別した。ボード値は連続測定の厳密な最大ではない。ローカル追加使用量は現時点約1.48GiBで、上限30GiB以内。**Kaggle GPUセッション消費は465.424秒＝7分45秒（約0.129時間）**で、累計8時間以内。GPU指定はP100だったが、実割当はUIでT4×2、実際の計算はCUDA device 0のTesla T4と確認した。最終集計は[resource_summary.json](outputs/wholebrain/resource_summary.json)、KaggleのGPU使用・終了状態は[kaggle_accounting.json](outputs/wholebrain/kaggle_accounting.json)を参照。

開始時のD空きは401.25GiB。Kaggle GPU枠は残り30時間・予約0・使用0、追加課金は無効だった。[通常Notebookの公開仕様](https://www.kaggle.com/docs/notebooks)はCPU/GPU最大12時間、保存出力20GBだが、今回の大会条件は9時間。実行時のCPU出力保存領域は約19.5GB空き、準備出力は約1.44GBだった。アカウント固有の追加ストレージ枠はAPIから取得できず不明。

初回CPU Notebookはkernelspec不足で起動直後に失敗し修正した。初回GPU特徴抽出は成功したが、エンコーダ計算以外の大部分が過剰な入力探索だった。修正版では大会DICOMをマウントせず前処理出力だけを使う。GPU各実行に30分timeoutを指定し、予約上限の合計で8時間を超えないガードを設けた。最新APIの表示だけでは過去の成功版が取得できなかったため、同一61検査キャッシュ（99.2MB）とImageNet重み（約103MB）を選択取得して主比較をローカルで完了。その後ブラウザで成功済みv1を特定し、SDKで出力を回収した。待機GPU版v2は0秒でキャンセルし、重複するCPU再現版v3も停止した。約570GBのDICOM一括取得はしていない。

Kaggle GPU特徴でも、再学習せず同じ学習済みOOF headを使ったmacro AUCは0.619574で一致した。全61検査の予測差は最大0.000318。実DICOMからKaggle CPUで推論した3公開例とローカルFP16結果の差は最大0.001345で、精度・ライブラリ・backend差による丸めの影響を確認した。詳細は[再現照合](outputs/wholebrain/replication_verification.json)。

## 全規模への推定（実測と区別）

**以下は全4,407検査・非公開約1,300検査での実測ではない。** 同じ系列選択・解像度・固定エンコーダを使う場合の換算値。

- 4,407検査の画像前処理：約2.15時間（Kaggle CPUの61検査平均から換算）。縮小画像キャッシュ約6.68GiB。
- 固定エンコーダ：約9.42分、実全脳伝播：約1.39分（ローカルRTX A1000の計測から換算）。生画像特徴約207MiB、回路特徴約34MiBを保存すれば再利用できる。
- readoutの4,407行負荷を58検査の反復で模擬したCPU計測は12分類器合計1.74秒。これは4,407件の独立症例での学習ではなく、収束反復数も実データでは変わる。3回/10回fitなら約5.2/17.4秒という単純換算。詳細は[負荷計測](outputs/wholebrain/readout_workload_benchmark.json)。
- 今回は固定特徴＋L-BFGSなので画像を複数epoch再計算しない。CNN微調整や内部26百万接続を更新する方式の全規模学習時間・メモリは未計測。
- **提出推論：約1,300検査をKaggle CPUで約3.26時間。** 3公開例で測った1件平均9.02秒＋起動処理から換算。各件処理が2倍なら約6.52時間という感度シナリオ。2倍は信頼区間ではない。長い系列、codec、ストレージI/O、CPU性能差で増える可能性がある。

約1,300件の隠しテストで9時間以内を実証したわけではない。コードはtest.csvの実際の件数を読み、3件に固定しない。8.5時間の内部監視、検査ごとの進捗保存、`submission.csv`形式検証を実装した。

## 成果物と再実行

- [再実行手順](WHOLEBRAIN_RUNBOOK.md)
- [事前に固定した比較仕様](docs/wholebrain_protocol.md)、[参照・回路仕様](docs/wholebrain_data_design.md)
- [比較詳細JSON](outputs/wholebrain/comparison/comparison.json)、[表CSV](outputs/wholebrain/comparison_summary.csv)、[各ラベルCSV](outputs/wholebrain/per_label_auc.csv)
- [全脳ID・注釈対応](data/connectome/male-cns-v1.0/node_mapping.csv.gz)、[入力・出力写像](outputs/wholebrain/comparison/circuit_mapping.csv.gz)、[全接続CSR](data/connectome/male-cns-v1.0/counts.npz)
- [Kaggle用オフライン推論Notebook](notebooks/wholebrain_inference/notebook.ipynb)、[生成確認済みsubmission.csv](outputs/wholebrain/kaggle_inference/submission.csv)
- [コード・重み成果物のSHA一覧](outputs/wholebrain/artifact_manifest.json)、[原表のSHA/URL一覧](data/connectome/male-cns-v1.0/source_manifest.json)

全てD:\RSNA配下。Kaggle Notebookは私有。大会への提出・一般公開は行っていない。元の環境確認用`outputs/submission.csv`は別物なので、今回の生成物は`outputs/wholebrain/kaggle_inference/submission.csv`を参照する。

## 未完了・制約と次の優先順位

全脳の実利用、比較評価、オフライン推論は成功。全4,407検査の画像学習、レポート由来ラベルの監査・導入、CNN微調整、内部接続の可塑化、隠しテスト評価、実大会スコアは未実施。残り4,349検査の公式ラベルは欠損であり、教師付き全規模学習の実施にはラベル作成・検証が必要。

次のGPU時間は、**検証検査を除外して監査済みレポート由来ラベルでMRIエンコーダを学習する実験**へ優先して使う。まずCPUで少数レポートの各ラベルの否定・不確実表現・欠損を照合し、公式と別の列・出典・confidenceで保存する。次に最大2 GPU時間の小規模CNN微調整を行い、公式ラベルだけの検証を維持してImageNet固定特徴から改善するか確認する。その同じ改善済み画像特徴を、通常readoutと全脳readoutへ渡して再比較する。

現在の結果では回路の拡大や伝播回数を増やす根拠は弱い。性能を優先するなら、最初に画像表現と教師ラベルの質を改善し、全脳構造の効果は同条件の迂回・シャッフル比較で引き続き分離して調べる。
