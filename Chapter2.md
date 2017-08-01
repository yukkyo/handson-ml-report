<!-- $theme: gaia -->

### Hands on Machine Learning with
###   Scikit-Learn &TensolFlow
## Chapter 2 
#### End-to-end Machine Learning project

###### Created by Yusuke FUJIMOTO

---

# はじめに

*  この資料は「[Hands-On Machine Learning with Scikit-Learn and TensorFlow - O'Reilly Media](http://shop.oreilly.com/product/0636920052289.do) 」
を読んだ際の（主にソースコードに関する）簡単な解説を残したものです。
*  全部を解説したわけではないので注意
*  あとは一部訳を間違えているかもしれないので信用しすぎないこと
*  余裕があればソースコード周りの背景知識もまとめたい

---

# Chapter 2 
# End-toend Machine Learning project

---

## Get the data
```python
import os
import tarfile
from six.moves import urllib

# ダウンロード元
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/"
			+ "ageron/handson-ml/master/"
# os 依存したパス構造を考慮
HOUSING_PATH = os.path.join("datasets", "housing")
# 実際にダウンロードするファイル名
HOUSING_URL = DOWNLOAD_ROOT +
		"datasets/housing/housing.tgz"
```
---

## Get the data（続き）
```python
def fetch_housing_data(housing_url=HOUSING_URL,
			housing_path=HOUSING_PATH):
    # ディレクトリが無かったら作る
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    # housing_url にアクセスして取ってきて tgz_path に配置
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    # tgz ファイルの展開
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
```
---
## Split train data and test data
```python
# For illustration only. Sklearn has train_test_split()
def split_train_test(data, test_ratio):
    # 0 から len(data) までの値をランダムに並べたもの
    shuffled_indx = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indx = shuffled_indices[:test_set_size]
    train_indx = shuffled_indices[test_set_size:]
    # python は一度に複数の値を返せる
    return data.iloc[train_indx], data.iloc[test_indx]
```
---
## Split train data and test data
```python
import hashlib

# この id のデータはテストデータか否かを返す
# 変数名が良くない -> is_test_data とかの方が良い
def test_set_check(identifier, test_ratio, hash):
    ident_int = np.int64(identifier) #  64bit整数に変換
    
    # 与えられた関数 hash を適用してダイジェスト（結果）表示
    hashed_obj = hash(ident_int).digest()
    
    # 最後の一文字を数値にしたもの。0から255までのばらけた数
    hashed_num = hashed_obj[-1]
    
    # 右辺の数より小さいか否か（True, False）を返す
    return hased_num < 256 * test_ratio
```
---

## Split train data and test data
```python
def split_train_test_by_id(data, test_ratio,
                           id_column, hash=hashlib.md5):
    ids = data[id_column] #  id とする列を指定
    # ids 内の各id に対して、テストデータか否かを求めた結果
    in_test = ids.apply(
                  lambda id_: test_set_check(
                                             id_,
                                             test_ratio,
                                             hash
                              )
              )
    # トレーニングデータとテストデータを返す
    return data.loc[~in_test], data.loc[in_test]
```
---

## Split train data and test data
*  apply : それぞれに関数を適用する関数
*  lambda : `lambda a: aの計算結果` は `a`を引数に `aの計算結果`を返す名前のない関数を表す
    * apply の中に入れる時などは便利
*  ~ : ビット反転。True なら False になる
    * `~np.array([True, False, False])` の結果は `np.array([False, True, True])` となる

---
## Split train data and test data
```python
from sklearn.model_selection \\
     import StratifiedShuffleSplit

# クロスバリデーション用オブジェクト
splitter = StratifiedShuffleSplit(
            n_splits=1,     #  何回再シャッフルするか
            test_size=0.2,  #  テストデータ比率
            random_state=42 #  乱数シード
        )
# splitter.split（データ、ラベル）で分割結果を返す
split_result = splitter.split(housing,
                              housing["income_cat"])

# 各ラベルについて、なるべく test と train 両方に割り振る
for train_index, test_index in split_result:
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
```

---
## Split train data and test data
言いたいこと：自作の train_test_split 関数（ランダムに振り分ける）だと内訳の比率が良くないよね
```python
def income_cat_proportions(data):
    # 各内訳の比率を返す
    return data["income_cat"].value_counts() / len(data)

train, test = train_test_split(housing, test_size=0.2,
                               random_state=42)

# cp = compare_props
cp = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test),
}).sort_index() #  index 順に並べ替え
```

---
### Prepare the data for ML algorithms
##### Imputer（欠損値の扱い方を決める）
```python
from sklearn.preprocessing import Imputer
impute = Imputer(strategy="median") #  mean,most_frequent
```

##### LabelEncoder（数値に変換）
```python
import numpy as np
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
sample = np.array(["a", "b", "c", "b", "c"])
sample2 = encoder.fit_transform(sample)
print(sample2)
# => array([0, 1, 2, 1, 2])
```

---
### Prepare the data for ML algorithms
##### OneHotEncoder（OneHot ベクトルに変換）
```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
sample3 = encoder.fit_transform(sample2.reshape(-1, 1))
print(sample3.toarray())
# [[ 1.  0.  0.]
#  [ 0.  1.  0.]
#  [ 0.  0.  1.]
#  [ 0.  1.  0.]
#  [ 0.  0.  1.]]
```
##### LabelBinarizer（0 か 1 の OntHot 表現に変換）
```python
from sklearn.preprocessing import LabelBinarizer
LabelBinarizer().fit_transform(sample) #  上と同じ書き方もok
```

---
### Prepare the data for ML algorithms
##### 平均家族数等を計算して付け足す
```python
from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, household_ix = \\
                                            3, 4, 5, 6
```
*  python では複数の変数に同時に代入できる

---
続き
```python
## 一部省略（クラス定義部分とか）
def transform(self, X, y=None):
    # 世帯あたりの部屋数、家族人数の計算
    rooms_per_household = \\
        X[:, rooms_ix] / X[:, household_ix]
    population_per_household = \\
        X[:, population_ix] / X[:, household_ix]
    
    # フラグに応じて追加したデータを返す
    if self.add_bedrooms_per_room:
        bedrooms_per_room = \\
            X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household,
                     population_per_household,
                     bedrooms_per_room]
    else:
        return np.c_[X, rooms_per_household,
                     population_per_household]
```
---
### Prepare the data for ML algorithms
##### 複数の（sklearn の）手続きを一連の流れにまとめる

```python
# 複数の処理や分類をまとめる
from sklearn.pipeline import Pipeline
# 標準化（だいたい平均 0 分散 1 が多い）するため
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)
```
