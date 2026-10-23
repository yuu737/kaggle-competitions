# Kaggriculture — 🥉 991 / 10,148

[大会ページ](https://www.kaggle.com/competitions/kaggriculture)

## 課題

農場経営シミュレーションの対戦型エージェント大会です。毎ターン観測を受け取り、作付け・収穫・家畜・雇用・売買の行動を返します。終了時の資金で勝敗が決まります。

## 解法

ルールベースの計画エージェント（[`solution/`](solution/)）。標準ライブラリだけで動きます。

- **役割ごとに分けた設計**: 観測、公式ルールの再現、投資計画、仕事の割り当て、購入を別のモジュールにした（[`solution/farm_model/`](solution/farm_model/)、設計は [`DESIGN.md`](solution/DESIGN.md)）。
- **公式との照合**: 市場120ケース、作業500ケース、両座席280ターンの状態遷移を公式の環境コードと突き合わせるテストを用意した。
- **判断の監査**: 対戦ログから「収穫し忘れ」「担当の決まらない仕事」などの判断ミスを洗い出し、再現ケースを回帰テストに固定した。
- 17候補・656試合の比較実験では、基本版を上回る改良は採用できなかった。

## ライセンス

`farm_model/rules.py`、`market.py`、`physics.py` のゲーム定数と挙動は、Kaggle公式の [kaggle-environments](https://github.com/Kaggle/kaggle-environments)（Apache License 2.0）に基づきます。[`NOTICE.txt`](solution/NOTICE.txt) と [`LICENSE.txt`](solution/LICENSE.txt) を参照してください。
