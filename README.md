# Financial Market Hypothesis Verification (FMHV)

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Author: Ryujiro Tsuji](https://img.shields.io/badge/Author-Ryujiro%20Tsuji-emerald.svg?style=flat-square)](https://tsujiryujiro.com)
[![Platform: Delver](https://img.shields.io/badge/Powered%20by-Delver-000000.svg?style=flat-square)](https://tsujiryujiro.com)

</div>

> **"In God we trust, all others must bring data."** > — *W. Edwards Deming*

本プロジェクトは、金融市場における多種多様なテクニカル指標、およびマーケットアノマリーの妥当性を、計算機科学と統計学の手法を用いて定量的に検証することを目的としたオープンソース・プロジェクトです。

---

## 1. 理念 (Philosophy)

現代のトレーディングシーンには、古くから伝わる「黄金比」や「ボリンジャーバンド」といった多種多様な通説（Hypothesis）が存在します。しかし、それらの多くは生存者バイアス、あるいは過学習（Overfitting）の産物であり、真の期待値に基づいた議論は極めて稀です。

本プロジェクトでは、**25年分（約6,500営業日）**に及ぶ時系列データをバックテストの基盤とし、現代のコンピューティングパワーを用いて、これらの手法が「真に統計的有意性を持つのか」、それとも「単なるホワイトノイズの解釈に過ぎないのか」を厳格に審判します。

---

## 2. 検証済みトピック (Verification Modules)

現在、以下のモジュールにおいて検証結果および再現用ソースコードを公開しています。

### A. Fibonacci Retracement Verification
フィボナッチ比率（38.2%, 61.8%等）における価格反発の優位性を検証。
- **Status:** `Completed`
- **Primary Finding:** 38.2%ラインにおける反発確率は、任意のランダムポイントにおける反発期待値の標準偏差内に収束。
- **Technical Report:** [フィボナッチの幻想と数学的実態](https://tsujiryujiro.com/blog/fibonacci-verification)

### B. Bollinger Bands Volatility Analysis
標準偏差に基づいたボラティリティ・ブレイクアウトの期待値推計。
- **Status:** `In Progress`
- **Key Metric:** 2σ超過後の平均回帰特性（Mean Reversion）の定量的分析。

---

## 3. ディレクトリ構造 (Project Structure)

```text
trade-verification/
├── scripts/
│   ├── fibonacci/      # フィボナッチ比率の統計的検証スクリプト
│   └── common/         # 時系列データ処理用共通ライブラリ
├── data/
│   └── processed/      # 検証済みの集計結果（CSV形式）
└── README.md
