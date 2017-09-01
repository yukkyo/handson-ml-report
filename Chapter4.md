<!-- $theme: gaia -->

### Hands on Machine Learning with
###   Scikit-Learn &TensolFlow
## Chapter 5
#### Support Vector Machines

###### Created by Yusuke FUJIMOTO

---

# はじめに

*  この資料は「[Hands-On Machine Learning with Scikit-Learn and TensorFlow - O'Reilly Media](http://shop.oreilly.com/product/0636920052289.do) 」
を読んだ際の（主にソースコードに関する）簡単な解説を残したものです。
*  全部を解説したわけではないので注意
*  余裕があればソースコード周りの背景知識もまとめたい
*  何かあったら yukkyo12221222@gmail.com まで
---

# Chapter 5
# Support Vector Machines

---

## 今回のポイント
* SVM（Support Vector Machine）は超メジャーな分類・回帰モデル
* ここまで扱った回帰モデルの損失関数（コスト関数、エラー関数）
$$
\mathrm{cost}(\bold x) = \sum^T_{i=i}(f(\bold x_i) - \bold y_i)^2
$$
* でもこれは、$f(\bold x_i)$ と $\bold y_i$ の距離を表しているわけではない

---
* 分類境界（ここでは超平面という）と各データの距離（マージン）に着目しているのが SVM の特徴
* あとサポートベクトルと呼ばれる一部のデータ（1つとは限らない）で識別境界が決定されるので、疎なデータに強い
* 識別境界 : $\omega^{\mathrm{T}} \bold x + b = 0$ 
  * 超平面からの距離 = $\omega$ の一次元ベクトルに写像しているだけ
### Large margin classification

* 特にコメントなし

---

### Sensitivity to feature scales
* 線形回帰モデルは、スケーリングをしないと精度に大きく影響した
* SVM ではスケールの変更前後で大きく変わらない（教科書確認）
* 
---

### Sensitivity to outliers
外れ値に対してどのように動作するのか確認する
* 外れ値が片方のクラスの散布に食い込んでいる場合
  * 線形分離不可能になっている
* 外れ値が片方のクラスの散布に非常に近い場合
  * 本来の境界より、片方のクラスに寄ったものになっている
---

### Large margin vs margin violations
マージンを大きくとるかマージン違反を許すか？？？（該当箇所を読む）
* 実問題では綺麗に線形分離できるとは限らない
  * 綺麗に分離できる場合しか境界を引けない（マージン違反を決して許さない） = ハードマージンSVM
  * マージン違反したデータに対してはペナルティを与えて、一応境界が引けるようにした SVM = ソフトマージンSVM
  * LinearSVC はソフトマージンSVM。実は。

---

* LinearSVCのオプションには `C` がある。これが正則化の強さ（線形回帰モデルにおける $\alpha$ の逆数にあたる）を表す。大きいほど強い。
* 小さいほどマージン違反を許し、大きいほど許さない
  * $C = \infty$ の場合にハードマージンSVMと等価
  * 先ほどの 「Sensitivity...」 では $C = 10^9$ となっており実質ハードマージンSVM
* $C=1$ の場合と $C=100$ の場合でどう変わるのか確認する。


---

### Non-linear classification
非線形な分離はできないのか？　→　できる
* higher_dimensions_plot : 一次元では線形分離ができなかったけど、二次元に写像したら線形分離できるようになった
* そういえばChapter 4 では polynomial feature で高次元特徴空間に写像できた！
  * polynomial svc
* 

---


### Regression

### Under the hood
内訳？？？

### Small weight vector results in a large margin

### Hinge loss

### Extra material


### Exercise solutions
