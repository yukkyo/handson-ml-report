<!--
$theme: default
prerender: true
page_number: true
$width: 12in
$size: 4:3
-->

### Hands on Machine Learning with
###   Scikit-Learn &TensolFlow
## Chapter 11
#### Deep Learning

###### Created by Yusuke FUJIMOTO

---

# はじめに

*  この資料は「[Hands-On Machine Learning with Scikit-Learn and TensorFlow - O'Reilly Media](http://shop.oreilly.com/product/0636920052289.do) 」
を読んだ際の（主にソースコードに関する）簡単な解説を残したもの
*  全部を解説したわけではないので注意
*  余裕があればソースコード周りの背景知識もまとめたい
*  何かあったら yukkyo12221222[@]gmail.com まで

---

# Chapter 11
# Training Deep Neural Nets


---

# ポイント

* ただ深いニューラルネットワークを作るだけだと問題がいっぱい
	* 勾配消失
	* 多大な計算量
* それらのためにいくつか工夫されている
* 11章では以下のポイントの詳細がある
	* 事前学習した層の再利用
	* より速い最適化
	* 正則化による過学習の抑制

---

# 1. Vanishing/Exploding Gradients Problem

## 1.1 Xavier and He Initialization

## 1.2 Nonsaturating Activation Functions

## 1.3 Batch Normalization

## 1.4 Gradient Clipping


# 2. Reusing Pretrained Layers

## 2.1 Reusing a TensorFlow Model

## 2.2 Reusing Models from Other Frameworks

## 2.3 Freezing the Lower Layers

## 2.4 Caching the Frozen Layers

## 2.5 Tweaking, Dropping, or Replacing the Upper Layers

## 2.6 Model Zoos

## 2.7 Unsupervised Pretraining

## 2.8 Pretraining on an Auxiliary Task

# 3. Faster Optimization

## 3.1 Momentum optimization

## 3.2 AdaGrad

## 3.3 RMSProp

## 3.4 Learning Rate Scheduling


# 4. Avoiding Overfitting Through Regularization

## 4.1 Early Stopping

## 4.2 $l_1$ and $l_2$ Regularization

## 4.3 Dropout

## 4.4 Max-Norm Regularization

## 4.5 Data Augmentation

# 5. Pratical Guidelines

# 6. Excercises