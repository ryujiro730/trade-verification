# Trade Verification Lab

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Author: Ryujiro Tsuji](https://img.shields.io/badge/Author-Ryujiro%20Tsuji-emerald.svg?style=flat-square)](https://tsujiryujiro.com)

</div>

本リポジトリは、ブログ「[delvertrade.com](https://delvertrade.com)」で公開しているトレード手法やテクニカル指標の検証コードを管理するプロジェクトです。

主観的な「通説」を排し、25年分（約6,500営業日）の時系列データを用いた統計的バックテストの結果を公開しています。

---

## 1. 検証プロジェクト一覧

現在、以下のモジュールにおいて検証結果および再現用ソースコードを公開しています。

### ■ フィボナッチ・リトレースメントの有効性
各フィボナッチ比率（38.2%, 61.8%等）における反発確率の統計的有意性を検証しています。
- **詳細記事:** [フィボナッチ検証のレポートを読む](https://tsujiryujiro.com/blog/fibonacci-verification)

### ■ ボリンジャーバンドのボラティリティ分析
標準偏差に基づいた逆張り・順張り戦略の期待値を推計しています。
- **詳細記事:** (近日公開予定)

---

## 2. ディレクトリ構造

```text
trade-verification/
├── scripts/
│   ├── fibonacci/      # フィボナッチ関連の検証スクリプト
│   ├── bollinger/      # ボリンジャーバンド関連の検証スクリプト
│   └── common/         # データ読み込み等の共通ユーティリティ
├── data/
│   └── processed/      # 集計済みの統計データ（CSV）
└── README.md

## 3.実行方法

datetime,OHLC構造のデータを用意してください。

# リポジトリのクローン
git clone [https://github.com/ryujiro730/trade-verification.git](https://github.com/ryujiro730/trade-verification.git)
cd trade-verification

# 依存ライブラリのインストール
pip install -r requirements.txt

# 検証スクリプトの実行
python scripts/fibonacci/research.py

## 4.著者情報

Ryujiro Tsuji

Website: tsujiryujiro.com

Tool: トレード検証プラットフォーム Delver 開発者

## 5.免責事項

本リポジトリの検証結果は情報の提供のみを目的としており、投資勧誘を意図するものではありません。取引に関する最終決定は、ご自身の判断で行ってください。

Copyright (c) 2024 Ryujiro Tsuji. Released under the MIT License.
