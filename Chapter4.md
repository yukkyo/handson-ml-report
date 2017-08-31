<!-- $theme: gaia -->

### Hands on Machine Learning with
###   Scikit-Learn &TensolFlow
## Chapter 4
#### Training Linear Models

###### Created by Yusuke FUJIMOTO

---

# はじめに

*  この資料は「[Hands-On Machine Learning with Scikit-Learn and TensorFlow - O'Reilly Media](http://shop.oreilly.com/product/0636920052289.do) 」
を読んだ際の（主にソースコードに関する）簡単な解説を残したものです。
*  全部を解説したわけではないので注意
*  余裕があればソースコード周りの背景知識もまとめたい
*  何かあったら yukkyo12221222@gmail.com まで
---

# Chapter 4
# Training Linear Models

---

## 今回のポイント
* これまで
  * 勝手に解を見つけてくれた（fit で）
  * もしくはパラメータを指定した範囲やランダムで当てはめて良いモデルを探した
* でも実際は・・・
  * 最適化したい関数はそんなに簡単ではない
  * パラメータが多すぎ、範囲広すぎ
  * もしくはデータが大きすぎて公式とかで解けない

---

* 今回は
  * より難しい関数（解の公式が無い場合等）等の解をどのように探すのか
  * 勾配法 (Gradient method)
    * 確率的勾配法 (Stochastic Gradient Descent)
    * ミニバッチ勾配法 (Mini-batch gradient descent)
---

* 他に扱うモデル
  * 多項回帰 (Polynomial regression)
  * 正規化モデル (Regularized models)
  * ロジスティック回帰 (Logistic regression)


---
### Linear regression using the Normal Equation

```python
# add x0 = 1 to each instance
X_b = np.c_[np.ones((100, 1)), X]
theta_best = \
    np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
```
上記のコードは以下の式を実装している。
まとめて掛け算に入れるため `np.c_` で繋いでいる。

$\theta_{best} = (\mathrm{X}_b^{\mathrm{T}} \times \mathrm{X}_b)^{-1} \times \mathrm{X}_b^{\mathrm{T}} \times y$

[線形回帰の Normal Equation（正規方程式）について](http://qiita.com/antimon2/items/ac1ebaed75ad58406b94)

---

### Linear regression using batch gradient descent
勾配法: ある地点からゴール（最適解）に向けて近づいていく方法の一つ

$\theta_{t + 1} = \theta_t - \eta \cdot \mathrm{gradients}(t)$

$\mathrm{gradients}(t) = \frac{1}{len(\mathrm{X}_b)} \cdot \mathrm{X}_b^{\mathrm{T}} \times \left(\mathrm{X}_b \times \theta_t  - y  \right)$

* $\theta_t$ : t 番目(iteration t 回目）のパラメータ
* $\eta$ : 更新率（ステップ幅とか）
* gradients : 勾配


---

## Stochastic Gradient Descent
通常の勾配法（gradient descent）の場合、訓練データ $\mathrm{X}$ を一度にすべて使って更新していた

それに対し、確率的勾配法（Stochastic Gradient Descent）では、 $\mathrm{X}$ の中ランダムに選んだデータ
$\mathrm{x}_i$ を使ってパラメータである $\theta$ を更新していく
* 局所最適解に落ちずらい
* 冗長な学習データがある場合勾配法より高速学習
* 学習データを収集しながら逐次敵に学習できる

---
## Mini-batch gradient descent
* batch がまとめて更新
* stochastic が 1 つずつ更新
* Mini-batch では小さいバッチサイズを決めてそのサイズ個のデータを使ってパラメータを更新していく
  * batch と stochastic の間をとったようなもの

---
## Polynomial regression
* 多項式（$x^n$ とかが入っている）の学習と予測について
* 今回は下記の式を扱う（ $e$ はランダムなエラー）
$$
\frac{1}{2}x^2 + x + 2 + e
$$
* `np.random.randn(m, n)` は m 行 n 列の乱数一覧を渡す（平均 0 分散 1）
---
#### `PolynomialFeatures` って何？
* 多項回帰式を求めるために、多項式次元に特徴ベクトルを写像している？
* 方針: 写像した後の空間で linear で回帰
* 高次の多項式だとクロスタームやらいっぱい出てきて書くのが大変
  * PolynomialFeature で特徴を作れば楽
* サンプルコードの例では、元のベクトルは 1 次元で、PolynomialFeatures の degree (設定した関数の次数)は 2 なので、$x -> (x, x^2)$ となっている
* TODO: 教科書の該当箇所を確認する

---
* もちろん予測したいときも、予測する特徴ベクトルを PolynomialFeature で写像する必要がある。

#### Q: plot_learning_curves は何をしている？


---

## Regularized models

## 


