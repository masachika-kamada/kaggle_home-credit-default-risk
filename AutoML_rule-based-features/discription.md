# Azure 自動ML

* LightGBM-with-Simple-Featuresで使用されていた特徴量生成には、専門的な知識が必要
* 知識がなくても生成できる特徴量のみで高いスコアを目指す
* 新たな特徴量生成はしない
* 値の集計をする際の処理を ['min', 'max', 'mean', 'size'] に固定
