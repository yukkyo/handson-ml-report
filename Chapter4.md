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

なんでこうなるかは教科書読もう（偏微分した結果）

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
## Mini-batch gradient descent
## Polynomial regression

## Regularized models

## 
