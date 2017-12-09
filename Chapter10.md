<!--
$theme: default
prerender: true
page_number: true
$width: 12in
$size: 4:3
-->

### Hands on Machine Learning with
###   Scikit-Learn &TensolFlow
## Chapter 10
#### Introdution to Artifical Neural Networks

###### Created by Yusuke FUJIMOTO

---

# はじめに

*  この資料は「[Hands-On Machine Learning with Scikit-Learn and TensorFlow - O'Reilly Media](http://shop.oreilly.com/product/0636920052289.do) 」
を読んだ際の（主にソースコードに関する）簡単な解説を残したもの
*  全部を解説したわけではないので注意
*  余裕があればソースコード周りの背景知識もまとめたい
*  何かあったら yukkyo12221222[@]gmail.com まで

---

# Chapter 10
# Introdution to Artifical Neural Networks


---

### 1. From Biological to Artificial Neurons

* NNの原型は、1943年に神経学者の *Warren McCulloch* と数学者の *Walter Pitts* によって提案された
* 1960年代と、1980年代に盛り上がる
  * しかし1990年代にSVMなどの手法が出てきて再び闇の時代に
* なぜ当時は廃れたのか？
  * 大量の学習データが必要
  * 当時から見るととてつもない計算力(CPU)が必要だった
  * 学習アルゴリズムが改善(1990年代より少ししか違わないけど、影響はとてつもなかった)
  * 当時謳われていたANNの理論的限界は、実はもっと良かった
  * 現在は 「関心を持つ → 発展する」という好循環に入っている

---

#### 1.1 Biological Neurons

* Cellbody(細胞体)
* axon(軸索)
* dendrite(樹状突起)
* ニューロンはある程度の情報が伝達されたら自身も発火（活動電位が発生）する

---

#### 1.2 Logical Computations with Neurons

* Artificial neuron(人口ニューロン)
  * *Warraen* と *Walter* が1943年に提案した非常にシンプルなニューロンモデル
  * 4通りの計算の方法がある
    * $C=A$、$C=A \land B$
    * $C=A \lor B$、$C=A \land \lnot B$

#### 1.3 The Perceptron

* 1957年に *Frank Rosenblatt* が提案したモデル
* LTU(Linear Threshold Unit) と呼ばれる前述のモデルと少し違うモデル
  * 論理値(True、False)の代わりに数が入るようになった
  * Output: $h_{\bold w}(\bold x) = \mathrm{step}(\bold w^t \cdot \bold x)$
  * それぞれの入力 $x_i$ に対し重み $w_i$ をかけて足し合わせたものをステップ関数に通す
* ステップ関数の例
  * $\mathrm{heaviside}(z) = 0 \space \mathrm{if} \space z < 0 \space \mathrm{else} \space 1$
  * * $\mathrm{sgn}(z) = -1 \space \mathrm{if} \space z < 0 \space$、$0 \space \mathrm{if} \space z = 0$、$1 \space \mathrm{if} \space z > 0$
* シングルLTU だけだと線形2クラス分類ができる

#### 1.3.1 パーセプトロンの学習法

$$
w_{i,j}^{(\mathrm{next} \space \mathrm{step})} = w_{i, j} + \eta(\hat{y}_j - y_j)x_i
$$

* $w_{i,j}$ : $i$ 番目の入力から、 $j$ 番目の出力までの重み
* $x_i$ : あるトレーニングデータの $i$ 番目の要素
* $\hat{y}_j, y_j$ : 予測ラベルと正解ラベル
* $\eta$ : 学習率
* この式は確率的勾配法に似ている

以下のようにして Python で試せる

```python3
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()
X = iris.data[:, (2, 3)]  # petal length, petal width
y = (iris.target == 0).astype(np.int)

per_clf = Perceptron(random_state=42)
per_clf.fit(X, y)

y_pred = per_clf.predict([[2, 0.5]])
```

#### 1.4 Multi-Layer Perceptron and Backpropagation(MLP)

* パーセプトロンをいくつか重ねている
  * → だから Multi-Layer
* 重ねると、Perceptron では解けなかった XOR問題などが解くことができる
  *  $(x,y)$ に対し、$x=y$ なら $False(0)$、$x \neq y$ なら $True(1)$ を返したい
  *  二次元平面上では線形分離できない
* 入力層(Input layer)と出力層(Output layer) 以外の中間にある層を隠れ層(Hidden layer)という
* 2つ以上隠れ層がある場合を、**deep neural network** という、らしい（本当か？）
* 出てきた当初はうまい学習法が見つけられなかった
  * 1986年に *D.E. Rumelhart* 達によって画期的な方法が提案された
  * backpropagation (誤差伝搬法)
  * 各ユニット間の重みについて勾配法を適用するイメージ

#### 1.4.1 Backpropagation (誤差伝搬法) 

主に以下のステップ

* ある訓練データに対し、予測を行う
* 予測誤差を求める
* 出力層から入力層にかけて、誤差を少なくするように重みを調整してゆく(逆向きに伝搬している)
* 誤差が減る (はず)


また、*Rumelhart* 達はこの方法が上手くできるように、activation function (活性化関数)を以下の２つのように変更した。

* ハイパボリックタンジェント
  * $tanh(z) = 2\sigma(2z) - 1$
  * $\sigma(z) = 1/(1 + \exp(-z))$ : ロジスティック関数
  * ロジスティック関数の値域を拡大したようなもの
  * 連続で、微分可能
* ReLU 関数
  * $\mathrm{ReLU}(z) = \max(0, z)$
  * 連続だが、$z=0$ で微分不可能
  * ただしプログラム上では以下のように定義可能
    * $$\displaystyle \frac{df}{dz} = 1 \space \mathrm{if} \space z \geq 0 \space \mathrm{else} \space 0$$
  * これを使うと上手くいくことが実験で確認された
  * 計算が速い
  * 何より、値の上限がない(これは11章で議論する。勾配消失に対処できる？)

MLP はしばしばクラス分類に用いられる。
多クラス分類で、かつ各クラスが排他 (0 から 9 の手書き画像とか) である場合は、
**softmax** 関数を出力層につなげることで作ることができる。

また、入力層から出力層にかけて、1方向のみに信号が進んでいくことから、
このような NN を
**feedforward neural network (FNN)** と呼ぶことがある。

※ 生物学的なニューロンは大体シグモイド活性化関数に似ていた。しかしANNにおいては ReLU 関数の方が上手くいっているので、これは生物学によるミスリードのケースの1つである。
→ なんでも似せれば良いというものではない (鳥の翼しかり)

### 2. Training an MLP with TensorFlow's High-Level API

Tensorflow で抽象度の高いモジュールが用意されている。

```python3
### データの用意
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/")
X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels.astype("int")
y_test = mnist.test.labels.astype("int")

### 学習
import tensorflow as tf

# 学習アルゴリズムのパラメータ(再現用)
config = tf.contrib.learn.RunConfig(tf_random_seed=42) # not shown in the config
# 特徴量の形式を定義している？
feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
# 学習器構築
dnn_clf = tf.contrib.learn.DNNClassifier(
                                         hidden_units=[300,100],       # 隠れ層を定義
                                         n_classes=10,                 # 出力クラス数
                                         feature_columns=feature_cols,           
                                         config=config
                                         )
# Scikit learn wrapper. Scikit-learn 風に扱える
# Sklearn のようにパイプラインに組み込めるかもしれない？
dnn_clf = tf.contrib.learn.SKCompat(dnn_clf) # if TensorFlow >= 1.1
# 学習
dnn_clf.fit(X_train, y_train, batch_size=50, steps=40000)

### 評価(分類実施)
from sklearn.metrics import accuracy_score

y_pred = dnn_clf.predict(X_test)
accuracy_score(y_test, y_pred['classes'])

# これでまとめて評価できる
dnn_clf.evaluate(X_test, y_test)
# => {'accuracy': 0.999..., 'global_step': 22222, 'loss': 0.0736...}
```

### 3. Training a DNN Using Plain Tensorflow

#### 3.1 Construction Phase

より細かく調整したい場合は、より低レベル(抽象度の低い)な APIを使う。

まずは1つのニューロン層を作ってみる。

```python3
import tensorflow as tf

n_inputs = 28*28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):  # 名前空間でパラメータを分ける
        # 各データの大きさ(特徴次元数)を確認。 shape[0] はデータの個数
        n_inputs = int(X.get_shape()[1])
        # W の初期化
        stddev = 2 / np.sqrt(n_inputs)
        # Tensorを正規分布かつ標準偏差の２倍までのランダムな値で初期化する
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="kernel")
        # バイアス
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        # Z を計算
        Z = tf.matmul(X, W) + b
        # 活性化関数に通す。未定義ならそのまま返す。
        if activation is not None:
            return activation(Z)
        else:
            return Z
```

上記のコード解説。

1. オプションで name scope を使うことができる。使うと TensorBoard で確認しやすくなる。
2. 特徴次元数を確認する
3. ある1データ内の値の標準偏差を使って、重み $\bold W$ を初期化。標準偏差を使うと収束が速くなるとのこと。
4. バイアス $\bold b$ を設定。層の中のニューロンの数だけまとめて定義。
5. $\bold Z = \bold X \cdot \bold W + \bold b$ でまとめて計算。
6. 最後に活性化関数に通す

次に以下のようにして DNN を構築。

```python3
with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, name="hidden1",
                           activation=tf.nn.relu)
    hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2",
                           activation=tf.nn.relu)
    logits = neuron_layer(hidden2, n_outputs, name="outputs")
```

※ TensorFlow の場合、`fully_connected` を使えば自身で `neuron_layer`などを用意する必要がない。

```python3
with tf.name_scope("dnn"):
    hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
    hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
    logits = fully_connected(hidden2, n_outputs, scope="outputs",
                             activation_fn=None)

```

そして以下のように誤差関数を定義する。

```python3
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
```

次に誤差関数を **GradientDescentOptimizer** によって最小化する部分を用意する。

```python3
learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
```

次に正解数を計算する部分を用意する。

```python3
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
```

最後に変数の初期化とパラメータ保存用のオブジェクトを用意する。

```python3
init = tf.global_variables_initializer()
saver = tf.train.Saver()
```

#### 3.2 Execution Phase


```python3
# 学習用パラメータ
n_epochs = 40
batch_size = 50

# 実際に計算を実施する。
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict={X: mnist.validation.images,
                                            y: mnist.validation.labels})
        print(epoch, "Train accuracy:", acc_train, "Val accuracy:", acc_val)

    save_path = saver.save(sess, "./my_model_final.ckpt")
```

#### 3.3 Using the Neural Network

学習させたパラメータを使って以下のように予測を行うことができる。

```python3
with tf.Session() as sess:
    saver.restore(sess, "./my_model_final.ckpt") # or better, use save_path
    X_new_scaled = mnist.test.images[:20]
    Z = logits.eval(feed_dict={X: X_new_scaled})
    y_pred = np.argmax(Z, axis=1)
```


### 4. Fine-Tuning Neural Network Hyperparameters

#### 4.1 Number of Hidden Layers



#### 4.2 Number of Neurons per Hidden Layer

#### 4.3 Activation Functions


### Excercise


### Appendix D (reverse-mode autodiff)

reverse-mode autodiff (逆向き自動微分？)について触れる。




##### 参考サイト
* [神経細胞 - Wikipedia](https://ja.wikipedia.org/wiki/%E7%A5%9E%E7%B5%8C%E7%B4%B0%E8%83%9E)
* [高卒でもわかる機械学習 (4) 誤差逆伝播法の前置き | 頭の中に思い浮かべた時には](http://hokuts.com/2016/05/27/pre-bp/)
* [Scikit-learn API - Keras Documentation](https://keras.io/ja/scikit-learn-api/)
* [[TF]TensorflowのAPIについて - Qiita](https://qiita.com/supersaiakujin/items/464cc053418e9a37fa7b)
* [TensorFlow理解のために柏木由紀さん顔特徴を調べてみた【中編】 - Qiita](https://qiita.com/FukuharaYohei/items/1192ca038b0052ef300f)