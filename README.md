# [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)

## 内容

* [LightGBM with Simple Features](https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features) のコードを元に上位解法のデータ処理を学ぶ
* 単一ファイルで実行されるkaggleノートブックのファイル分けを行い、他のコンペティションでも使用できるような形に落とし込む
* 専門知識がなくても高いスコアを得ることができないか、Azureの自動MLを使用して実験

## 結果

* ノートブックのkfold_lightgbmで学習
  * LightGBM with Simple Featuresオリジナルの特徴量を使用
    * private: 0.79023 / public: 0.79136
  * 新たな特徴量を使用しない制約あり
    * private: 0.78445 / public: 0.78173

* Azure自動ML
  * LightGBM with Simple Featuresオリジナルの特徴量を使用
    * private: 0.72382 / public: 0.72553
  * 新たな特徴量を使用しない制約あり
    * private: 0.72756 / public: 0.73050
