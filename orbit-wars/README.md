# Orbit Wars — 748 / 4,730

[大会ページ](https://www.kaggle.com/competitions/orbit-wars)

## 課題

太陽の周りを回る惑星を艦隊で奪い合う、2人または4人対戦のリアルタイム戦略ゲームのエージェント大会です。惑星は公転し、彗星が横切り、太陽を通る艦隊は消滅します。500ターン後の艦数で勝敗が決まります。

## 解法

ルールベースのエージェント（[`main.py`](main.py)、[`my_orbit/`](my_orbit/)）。

- **シミュレータ**（[`my_orbit/simulator.py`](my_orbit/simulator.py)）: 惑星の公転と艦隊の移動を先読みし、到着時点の戦力を予測する。
- **候補の生成と評価**: 攻撃先の候補を作り（`candidates.py`）、得点を付けて（`scoring.py`）、安全性を確認して（`safety.py`）選ぶ。
- **評価**: 過去の自分の版と対戦させ、2人戦・4人戦の全座席で勝率を比べる（[`evaluate.py`](evaluate.py)）。
