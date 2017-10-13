<!--
$theme: gaia
prerender: true
page_number: true
$width: 12in
$size: 4:3
-->

### Hands on Machine Learning with
###   Scikit-Learn &TensolFlow
## Chapter 8
#### Dimensionality Reduction

###### Created by Yusuke FUJIMOTO

---

# はじめに

*  この資料は「[Hands-On Machine Learning with Scikit-Learn and TensorFlow - O'Reilly Media](http://shop.oreilly.com/product/0636920052289.do) 」
を読んだ際の（主にソースコードに関する）簡単な解説を残したものです。
*  全部を解説したわけではないので注意
*  余裕があればソースコード周りの背景知識もまとめたい
*  何かあったら yukkyo12221222@gmail.com まで

---

# Chapter 8
# Dimensionality Reduction

---

#### ポイント

* データの特徴次元（特徴数）が大きいと、学習が遅いだけでなく良い結果が得られない
   * 次元の呪い
* 次元削減すれば対処できる場合がある
* PCA は分散が大きい順に軸を取り出す(線形な座標変換をする)
* 第 $i$ 軸は第 $i-1$ 軸と直交と定義すると逐次的に計算できる

---

#### The Curse of Dimensionality

* 日常生活は 3 次元だから高次元イメージしづらい
* $1 \times 1$ の四角形(2次元)を考える
   * ランダムな点を指定して、点が四角形の枠から $0.001$ 以内にある可能性は、
   * $1 - (1 - 0.001 \times 2)^2 = 0.003999\cdots$、約 0.4%
* $10,000$ 次元の四角形だと、
   * $1 - (1 - 0.001 \times 2)^{10000} = 0.9999\cdots$、99.999999% 以上になる ⇒ ほとんどのデータは枠の付近にある

---

* これらの可能性から、高次元にあるデータは **スパース**であるリスクがある
* どのデータ間でも距離が大きすぎる
* Overfitting しやすい
   * 解決法の一つ: Training Data 増やす
   * 十分な密度になるくらいデータを増やす
   * 100 次元の空間だと、もし各データの（最も近いデータ？との）距離が平均して0.1 以内にあるためには、観測可能な宇宙内の原子の数よりもデータが必要らしい？　⇒　たくさんデータが必要

---

### Main Approaches for Dimensionality Reduction

* Projection (低次元への写像)
  * ある低次元の空間（2次元だと面、1次元だと線）に、各データを写す
* Manifold Learning (多様体学習)
  * $d$ 次元多様体: 元は $d$ 次元の空間（線とか面）を、より大きな$n(n > d)$次元上で曲げたりひねったりしたもの
  * 例: Swiss roll は、2次元の空間（面）を、3次元でくるくる巻いた形になっている
---

#### Approach 2 : Manifold Learning

* Swiss roll データの場合、局所的にみると2次元の面に似ている
* MNIST データの場合、各文字画像は、線が繋がってできている、境界線は白、大体中央に文字がある、などの共通点がある ⇒ 「文字画像を生成する自由度 << 全くランダムな画像を作る自由度 」
* これらの制約はデータセットをより低次元に圧縮する傾向がある

---

* 多様体である仮定を置くこと ⇒ 低次元で表現するとより簡単になることを仮定している
  * いつもそうとは限らない（テキスト図参照）
  * ⇒ モデルのトレーニング前に次元削減を行うことで、**学習は速くなるが、より良い精度やよりシンプルに解けるようになるとは限らない**

---

## PCA

* PCA: Principal Component Analysis (主成分分析)
* 最初にデータの最も近い超平面を特定し、その超平面にデータを写像する


#### PCA: Preserving the Variance

* 二次元データの場合: どの直線（方向）に写像すると、最も分散が大きくなるのかをテキストの図で確認する
* ⇒ 各データの写像先データの距離（超平面との距離みたいなもの）の平均二乗誤差が最も小さくなる超平面を探していることと等価

---

#### PCA: Principal Components

* orthogonal: 直交
* ある軸（主成分）を探す際に、それまでに探した主成分すべてと直交するような軸を探す
* 

#### PCA: Projecting Down to d Dimensions

* SVD(特異値分解) してる
* PCA では各点が原点の中心にあると仮定している ⇒ 各データについて、そのデータの平均値が引いてあることを確認する ⇒ Scikit-laern は勝手に引いているから気にしなくてよい

---

#### PCA: Using Scikit-Learn


#### PCA: Explainged Variance Ratio

* ある主成分が、データ全体の分散のうちどれくらいを占めるのかを表す。
* 主成分をどこまで残すのかを検討する際に参考にする

---

#### PCA: Choosing the Right Number of Dimensions

#### PCA: PCA for Compression


---

#### PCA: Incremental PCA


#### PCA: Randomized PCA

---

### Kernel PCA

#### Kernel PCA: Selecting a Kernel and Tuning Hyperparameters

aaa

---

### LLE

* 多様体学習の一つ
* 多様体: 元は線形である空間（面とか線とか）を曲げたりねじったりして高次元上に表される空間
* 局所的な位置関係を保存して低次元に写像しようとしている
* 

---

### LLEの詳細

1.  あるデータに $\bold x$ に対して $K$ 最近傍(最も近い $K$ 個のデータのこと)を選出する
    * 最近傍で得られたデータ集合を $\kappa$ とする
2.  $\kappa$ に属するデータ集合 $\bold x_j \in \kappa$ の線形結合を考える。これが小さくなれば、近傍データで $\bold x$ を表現できていることになる

$$
\left| \bold x - \sum_{\bold x_j \in \kappa}w_j \bold x_j \right|^2
$$


---

3. 全てのデータ $\bold x_i$ に対して上記の関数を作る

$$
\left| \bold x_i - \sum_{\bold x_j \in \kappa^{(i)}}w_j^{(i)} \bold x_j \right|^2
$$


---


### Other Dimensionality Reduction Techniques

* Multidimensional Scaling (MDS)
* Isomap
* t-distributed
* Linear Discriminant Analysys (LDA)

---

##### Isomap

* 非線形次元削減手法の一つ
* K近傍グラフを用いて多様体上の測地線距離を（近似的に）求め、多次元尺度構成法を用いて近時的にユークリッドな低次元空間に射影する


---

#### 補足: 共分散

ある $n$ 次元のデータ $x$ と $y$ の間の共分散

$$
s_{xy} = \frac{1}{n}\sum^n_{i=1}(x_i - \bar{x})(y_i - \bar{y})
$$

* $n$ : 各データの次元
* $\bar{x}$ : $x$ の平均
* $\bar{y}$ : $y$ の平均

---

#### 補足: 共分散

python だと下記のように求められる

```python
x = np.array([1, 2, 4, 5])
y = np.array([5, 6, 8, 9])
x_bar = x.mean()
y_bar = y.mean()
s = (x - x_bar) @ (y - y_bar)
s / x.shape[0]
```

---

### Exercises



##### 参考サイト
* [PCAの最終形態GPLVMの解説](https://www.slideshare.net/antiplastics/pcagplvm)
* [Rでisomap（多様体学習のはなし）](https://www.slideshare.net/kohta/risomap)
* [【多様体学習】LLEとちょっとT-SNE - HELLO CYBERNETICS](http://s0sem0y.hatenablog.com/entry/2017/07/06/133450)