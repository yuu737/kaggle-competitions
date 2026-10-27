# RSNA Knee Abnormality Detection — 402 / 6,300（メダル確定待ち）

[大会ページ](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)

## 課題

膝のMRI（DICOM、全体で約570GB）から、複数の異常（関節炎、滑膜炎、ベーカー嚢腫など）の有無を予測する医用画像の大会です。

## やったこと

- **状況の分析**: 公開順位表を全件取得すると、0.958のチームが1,000以上あり、ほぼ同じ公開ノートブックのコピーだった。その塊の中の順位には意味がなく、非公開順位で銀に残れる確率は1〜2割と見積もった。
- **ラベルの作り直し**: 公開議論で効果が報告されていた、out-of-foldの予測を元ラベルに混ぜる疑似ラベルで、自前のCoAtNetを学習した。
- **ブレンド**: 公開最良のブレンドに自前モデルを加える形で提出した。公開順位では、足すと少し下がる結果だった（0.958 → 0.956〜0.957）。
- **試して効かなかったこと**: ショウジョウバエの全脳コネクトームから作った視覚モデルの特徴を加える実験。計算とオフライン推論までは完了したが、精度の有意な向上は確認できなかった。

## コード

- [`modal/train_student.py`](modal/train_student.py): 疑似ラベルで自前のCoAtNet（352px）を学習するスクリプト。クラウドGPUの [Modal](https://modal.com/) で実行した。学習器はnartaaさんの公開コード（Apache-2.0）をそのまま使い、データの差し替えと解像度の設定だけを加えている。ラベルは「公開モデルの軟ラベル」と「自分の4分割OOF予測を補正したもの」を半分ずつ混ぜた
- [`scripts/prepare_public_labels.py`](scripts/prepare_public_labels.py)、[`make_stage2.py`](scripts/make_stage2.py): ラベルと2段目の入力の準備
- 全脳コネクトームの実験: [`trainable_wholebrain.py`](scripts/trainable_wholebrain.py)、[`train_wholebrain_adaptive.py`](scripts/train_wholebrain_adaptive.py)、[`flyvis_wholebrain_prior.py`](scripts/flyvis_wholebrain_prior.py) ほか。結果は [`reports/`](reports/)

## Claude Codeの使い方

Claude Codeで進めた。「今のままでは銀メダルに残れない。計画を立て直せ」と指示し、Kaggle APIで順位表の全件、上位11本のノートブック、議論110件を取得させ、そこから計画を立て直させた（[計画書](reports/PLAN_V2_20261019.md)）。
