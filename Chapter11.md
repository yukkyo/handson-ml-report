<!-- コード隠す用
<details><summary>サンプルコード</summary><div>
</details></div>
-->

# はじめに

*  この資料は「[Hands-On Machine Learning with Scikit-Learn and TensorFlow - O'Reilly Media](http://shop.oreilly.com/product/0636920052289.do) 」
の Chapter 11 を読んだときのまとめです
*  元コードは https://github.com/ageron/handson-ml にあります
*  筆者の英語力により誤訳の可能性あります。気づいた方は教えてくださると幸いです
*  気になる人は買って一緒に読み進めましょう(興味ある人大歓迎です)

# 対象

* Python やそのライブラリ(主に TensorFlow)触ったことある人
* TensolFlow での具体的な実装知りたい人
* 深いニューラルネットの精度向上につながる細かいテクニックが知りたい人

# まとめ

* ただ深いニューラルネットワークを作るだけだと問題がいっぱい
	* 勾配消失
	* 多大な計算量
* それらを改善するためにいくつか工夫をいれる
	* 活性化関数を変える
	* 重みの初期化
	* 事前学習した層の再利用
	* より速い最適化
	* 正則化による過学習の抑制

上記について概要に加え TensorFlow のコードに落とし込める人は読まなくて大丈夫です。

# 前準備


必要なライブラリや関数を定義しておきます。
`reset_graph` は検証環境などでは必須な気がします。

<details><summary>サンプルコード</summary><div>

```Python
import tensorflow as tf
import numpy as np

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# to make this notebook's output stable across run
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# Loading Data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")
```

</details></div>


データのロードが上手く行けば下記のような結果が出ます

<details><summary>結果</summary><div>

```text
Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
Extracting /tmp/data/train-images-idx3-ubyte.gz
Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Extracting /tmp/data/train-labels-idx1-ubyte.gz
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Extracting /tmp/data/t10k-images-idx3-ubyte.gz
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Extracting /tmp/data/t10k-labels-idx1-ubyte.gz
```

</details></div>

# 1. Vanishing/Exploding Gradients Problem(勾配消失/爆発問題)

* 勾配消失して重みが変わらなくなる => 勾配消失問題
* 逆に勾配が爆発する場合もある => 勾配爆発問題
* Back probagation 時に重みが層ごとに掛け算されていくからなんとなくイメージできそう
* これも一因となり、 2010年頃まで Deep neural network は避けられてきた
* 2010 年に [[Xavier Glorot and Yoshua Bengio, 2010]](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) によって進捗が出た
	* この界隈で Bengio さん知らないって言ったら Deep Learner に刺されるのでお気をつけて


## 1.1 Xavier and He initialization(He initialization)

* またの名を `Glorot initialization`
* 上記の論文で提示されていた初期化方法
* これのおかげで従来より学習が安定
* 正規分布か、一様分布から初期値をランダム生成する
* 対象の活性化関数はロジスティック関数だが、[[Microsoft Research, 2015]](https://arxiv.org/pdf/1502.01852v1.pdf)では他の活性化関数(ReLU)用の初期パラメータも発表されている

下記に初期化用パラメータをまとめました。

* 正規分布の平均は $\mu = 0$ です。
* $n_{\mathrm{inputs}}$、$n_{\mathrm{outputs}}$ は入力、出力の接続数です。

|活性化関数  | Uniform distribution(一様分布) $[-r, r]$  | Normal distribution(正規分布)  |
|---|---|---|
| Logistic  | $\displaystyle r = \sqrt{\frac{6}{n_{\mathrm{inputs}} + n_{\mathrm{outputs}}}}$  | $\displaystyle \sigma = \sqrt{\frac{2}{n_{\mathrm{inputs}} + n_{\mathrm{outputs}}}}$   |
| Hyperbolic tangent  | $\displaystyle r = 4\sqrt{\frac{6}{n_{\mathrm{inputs}} + n_{\mathrm{outputs}}}}$  | $\displaystyle \sigma = 4\sqrt{\frac{2}{n_{\mathrm{inputs}} + n_{\mathrm{outputs}}}}$   |
| ReLU(ELU などの派生含む) | $\displaystyle r = \sqrt{2}\sqrt{\frac{6}{n_{\mathrm{inputs}} + n_{\mathrm{outputs}}}}$  | $\displaystyle \sigma = \sqrt{2}\sqrt{\frac{2}{n_{\mathrm{inputs}} + n_{\mathrm{outputs}}}}$   |


ちなみに TensorFlow だと以下のようにして Xavier initialization(with a uniform distribution) を使うことができます。

```python
he_init = tf.contrib.layers.variance_scaling_initializer()
hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu,
                          kernel_initializer=he_init, name="hidden1")
```



## 1.2 Nonsaturating Activation Functions(不飽和活性化関数)

### 1.2.1 LeakyReLU
* 上記の 2010年の論文だとロジスティック関数を対象にしていた
* ロジスティック関数は入力が大きいときや小さいときは $1$ や $0$ 付近で値がほとんど変わらない(飽和している)
* 他の活性化関数(ReLU など)の方が上手くいくかもしれない
* しかし通常の ReLU( $\mathrm{ReLU}(z) = \max(0, z)$ ) の場合は出力が 0 になり出力が消えてしまう問題がある(**dying ReLUs**)
* そこで出力の下限を 0 ではなく $\alpha z$($\alpha$ は小さな定数)にする
$\mathrm{LeakyReLU_{\alpha}(z) = \max(\alpha z, z)}$ などが登場
	* $\alpha$ は 0.01 や 0.2 などの定数などもあるし、トレーニング時にランダムにピックアップする **randomized leaky ReLU** もある
	* また $\alpha$ も学習対象として backprobagation 時に学習する
	**parametric leaky ReLU(PReLU)** もある
* 0 にはしないことで死なないようにした => また再起する可能性を得た


<!--
できれば図を入れる
-->

### 1.2.2 ELU

* [[Djork-Arn ́e, 2015]](https://arxiv.org/pdf/1511.07289v5.pdf)
* **exponential linear unit(ELU)**
* **leaky ReLU** は大量画像データでは通常の **ReLU** を大幅に上回る性能だが、少量データセットでは過学習のリスクがあった。
* **ELU** は **ReLU** やその派生系より学習時間が短く性能も良かった

```math
\mathrm{ELU}_{\alpha}(z) = \left\{
\begin{array}{ll}
\alpha(\exp(z) - 1) & (z \lt 0) \\
z & (z \geq 0)
\end{array}
\right.
```

**ReLU** と **ELU** の主な違いは以下の通りです。

* $z < 0$ で負値を返すので、ユニットの出力の平均が 0 付近になる => 勾配消失問題に対処
* $z < 0$ でも非 0 の勾配となる => 'dying unit issue' の回避
* 関数が連続 => 0 の前後で勾配の落差がない => 勾配法の学習スピード向上

**ELU** は $\exp$ 計算が入るため、**ReLU** 単体より計算が遅いですがその分収束が早いです。

=> しかし Test 時は **ELUの方が遅いです**

※ 一般には、**ELU** > **leaky ReLU**(とその派生) > **ReLU** > $\tanh$ > ロジスティック(おすすめ順)とのこと。しかし実行時間が気になるなら **ReLU**、計算時間に余裕があるなら **RReLU(randomized ReLU)** で過学習を抑制したり、大量データがあるなら **PReLU** を採用するなどの選択肢もあります。


## 1.3 Batch Normalization

* 入力データを小分けしただけのバッチ法とは別なので注意
* 上記の **He initialization & ELU** によって学習開始時の勾配消失問題は大幅に軽減されました。

* しかし学習途中まで保証してはいませんでした。
* **Internal Covariate Shift problem** 問題
	* 学習中に各層の入力の分布が変わってしまうこと
	* 要調査
* [[Sergey Ioffe &  Christian Szegedy, 2015]](https://goo.gl/gA4GSP) では上記の問題に対応するためにバッチ学習法を提案
* これにより勾配消失がさらに減ったとのこと
* 予測時にはテストデータ全体の平均と標準偏差を用いてテストデータにも正規化を行う
	* その分予測に時間がかかる
	* もし予測時間を短縮したい場合は、**ELU + He initialization** でのパフォーマンスも予め計測して検討すると良いらしいです

バッチ正規化アルゴリズム(Batch Normalization, 以下BN)の式は以下の通りです。

```math
\displaystyle
\begin{array}{ll}
\mu_B &=& \frac{1}{m_B} \sum_{i=1}^{m_B} \boldsymbol{x}^{(i)} \\
\\
\sigma_B^2 &=& \frac{1}{m_B}\sum_{i=1}^{m_B}(\boldsymbol{x}^{(i)} - \mu_B)^2 \\
\\
\hat{\boldsymbol{x}}^{(i)} &=& \frac{\boldsymbol{x}^{(i)} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \\
\\
\boldsymbol{x}^{(i)} &=& \gamma \hat{\boldsymbol{x}}^{(i)} + \beta
\end{array}
```

各パラメータは以下の通り

|パラメータ|内訳|
|:----------|:-------|
| $\mu_B$ | 平均、ミニバッチ $B$ 毎に計算(評価)する |
| $\sigma_B$ | 標準偏差、これもミニバッチ毎に計算する |
| $m_B$ | ミニバッチ内のデータ数(特徴ベクトル数) |
| $\hat{\boldsymbol{x}}^{(i)}$ | ゼロ平均化と標準化した入力('zero-centered and normalized' と書いてあったので、標準化とゼロ平均化は分けている？) |
| $\gamma$ |層のスケーリング用パラメータ |
| $\beta$ | 層のオフセットパラメータ |
| $\epsilon$ | 0除算を防ぐための小さい値($10^{-3}$ など)、**smoothing term** とも呼ばれる |
| $\boldsymbol{x}^{(i)}$ | BN 操作による出力 |

また BN法を適用するにあたり、モーメント

### 1.3.1 TensolFlow での実装

#### 1.3.1.1 バッチ使わない版

<details><summary>サンプルコード</summary><div>

```python
reset_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

# ネットワーク定義(全然 Deep に感じられない人はもっと中間層入れましょう)
with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=leaky_relu, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=leaky_relu, name="hidden2")
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

# 損失関数定義(最小化したい対象)
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

# 最適化方法選択
learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

# 学習結果評価
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()  # 初期化用のおまじない
saver = tf.train.Saver()  # 学習モデル保存用

n_epochs = 40
batch_size = 50
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        # ここでも Batch と書いてありますが、あくまでデータを小分けにしただけなので注意
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if epoch % 5 == 0:
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict={X: mnist.validation.images, y: mnist.validation.labels})
            print(epoch, "Batch accuracy:", acc_train, "Validation accuracy:", acc_test)

    save_path = saver.save(sess, "./my_model_final.ckpt")
```
</div></details>

#### 1.3.1.2 He initialization + ELU + バッチ法適用

* なおもっと深くしないと ELU + バッチ法による改善は見られないそうです・・・

<details><summary>サンプルコード</summary><div>

```Python
# 固定引数を省略するための関数。便利です
from functools import partial
reset_graph()

batch_norm_momentum = 0.9

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")
training = tf.placeholder_with_default(False, shape=(), name='training')

with tf.name_scope("dnn"):
    he_init = tf.contrib.layers.variance_scaling_initializer()

    my_batch_norm_layer = partial(tf.layers.batch_normalization,
                                  training=training,
                                  momentum=batch_norm_momentum)

    my_dense_layer = partial(tf.layers.dense,
		                         kernel_initializer=he_init)

    # 各層の活性化関数の適用前に正規化を行う
    hidden1 = my_dense_layer(X, n_hidden1, name="hidden1")
    bn1 = tf.nn.elu(my_batch_norm_layer(hidden1))  # ELU
    hidden2 = my_dense_layer(bn1, n_hidden2, name="hidden2")
    bn2 = tf.nn.elu(my_batch_norm_layer(hidden2))  # ELU
    logits_before_bn = my_dense_layer(bn2, n_outputs, name="outputs")
    logits = my_batch_norm_layer(logits_before_bn)

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 20
batch_size = 200

# これは何？ => バッチ法における 平均や標準偏差も更新するための操作
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run([training_op, extra_update_ops],
                     feed_dict={training: True, X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images,
                                                y: mnist.test.labels})
        print(epoch, "Test accuracy:", accuracy_val)

    save_path = saver.save(sess, "./my_model_final.ckpt")
```

</details></div>


## 1.4 Gradient Clipping

* exploding gradients problem(勾配爆発問題)に効果のあるテクニック
* 特に RNN(Recurrent Neural Network) で効果があるらしい
* [参考スライド](https://www.slideshare.net/tsubosaka/deeplearning-60419659)

> 勾配が大きくならないように、gradient の値を計算した後に値が閾値を超えていたら修正する

<details><summary>サンプルコード</summary><div>

```Python
# 適用しないとき
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

# Gradient Clipping を適用するとき
threshold = 1.0
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(loss)  # 勾配計算
# tf.clip_by_value で閾値より大きいなら閾値に修正している
capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var)
              for grad, var in grads_and_vars]  # 閾値より大きい値なら修正
training_op = optimizer.apply_gradients(capped_gvs)
```

</details></div>


# 2. Reusing Pretrained Layers

* 巨大な DNN を1から学習するのはあまり良くない
* 他の類似タスクで学習した結果(低層)を用いる
	* **transfer learning(転移学習)** とも呼ばれる
* 学習時間の短縮や学習データの少量化にも効果がある
* 例
	* 入力層、隠れ層(1~5)、出力層からなる DNN を事前に学習する
	* 上記ネットワーウの入力層、隠れ層(1~3) をそのまま用いる
		* ただし**隠れ層1** と **隠れ層2** の**重みは固定**

<!--
できればイメージ図を参照したい
-->

## 2.1 Reusing a TensorFlow Model

### 2.1.1 まるまる再利用する場合

これまでのモデルとパラメータをそのまま使う場合は `tf.train.import_meta_graph()` と `saver.restore()` でそのまま読み込むことができます

<details><summary>サンプルコード</summary><div>

```Python
saver = tf.train.import_meta_graph("./my_model_final.ckpt.meta")

with tf.Session() as sess:
    saver.restore(sess, "./my_model_final.ckpt")  # これで先程の学習で保存したモデルとパラメータを読み込み
		# 以下で追加学習
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images,
                                                y: mnist.test.labels})
        print(epoch, "Test accuracy:", accuracy_val)

    save_path = saver.save(sess, "./my_new_model_final.ckpt")    
```

</details></div>



### 2.1.2 一部の層のみ使いたい場合

例として、上記で述べたように、隠れ層を5つ使った DNN を構築し学習させ、そのうちの隠れ層1から3を再利用してみます。

まずは隠れ層が5つある DNN を構築します。
ついでに上記にあった **Gradient Clipping** を使ってみてます。

<details><summary>サンプルコード</summary><div>

```Python
reset_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_hidden2 = 50
n_hidden3 = 50
n_hidden4 = 50
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
    hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name="hidden3")
    hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="hidden4")
    hidden5 = tf.layers.dense(hidden4, n_hidden5, activation=tf.nn.relu, name="hidden5")
    logits = tf.layers.dense(hidden5, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

learning_rate = 0.01
threshold = 1.0

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(loss)
capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var)
              for grad, var in grads_and_vars]
training_op = optimizer.apply_gradients(capped_gvs)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "./my_model_final.ckpt")

    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images,
                                                y: mnist.test.labels})
        print(epoch, "Test accuracy:", accuracy_val)

    save_path = saver.save(sess, "./my_new_model_final.ckpt")
```

</details></div>

次に上記の学習結果から隠れ層1~3を再利用、隠れ層4と出力層を新しく用意します。
ポイントは `tf.get_default_graph().get_tensor_by_name()` を使って前回の結果からグラフを取得することですが、その際に

`hidden3 = tf.get_default_graph().get_tensor_by_name("dnn/hidden4/Relu:0")`
のように、 `hidden4` まで指定していることがポイントです。


<details><summary>サンプルコード</summary><div>

```Python
reset_graph()

n_hidden4 = 20  # new layer
n_outputs = 10  # new layer

saver = tf.train.import_meta_graph("./my_model_final.ckpt.meta")

X = tf.get_default_graph().get_tensor_by_name("X:0")
y = tf.get_default_graph().get_tensor_by_name("y:0")

hidden3 = tf.get_default_graph().get_tensor_by_name("dnn/hidden4/Relu:0")

new_hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="new_hidden4")
new_logits = tf.layers.dense(new_hidden4, n_outputs, name="new_outputs")

with tf.name_scope("new_loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=new_logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("new_eval"):
    correct = tf.nn.in_top_k(new_logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

with tf.name_scope("new_train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
new_saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()
    saver.restore(sess, "./my_model_final.ckpt")

    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images,
                                                y: mnist.test.labels})
        print(epoch, "Test accuracy:", accuracy_val)

    save_path = new_saver.save(sess, "./my_new_model_final.ckpt")
```

</details></div>

## 2.2 Reusing Models from Other Frameworks

ここでは他のフレームワーク(自作含む)で求めたパラメータで Tensorflow のネットワークを構築する方法を紹介します。
(互換性のあるフレームワークなら用意してある関数使ったほうが良いと思います)

今回は互換性が全くない状態を想定します。
事前に以下のようにして他のフレームワーク等からパラメータを抽出してある状態を想定しています。


<details><summary>共通部分</summary><div>

```Python
reset_graph()

n_inputs = 2
n_hidden1 = 3

original_w = [[1., 2., 3.], [4., 5., 6.]]  # weight, 他のフレームワークから持ってきたパラメータ
original_b = [7., 8., 9.]                  # bias, 他のフレームワークから持ってきたパラメータ

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
# [...] 以下に自分で組んだ他の層が定義されているとする
```

</div></details>


上記で組んだモデル内の隠れ層1のパラメータに `original_w` と `original_b` を適用します。


### 2.2.1 (比較的)簡潔な方法

ここでの "kernel" は "weights" のことを指しています。
おおまかな流れは以下の通りです。

* `graph.get_operation_by_name` で該当の層の kernel(weights) や bias の Operation オブジェクトをとってくる
* 各 Operation オブジェクトへの入力となる Tensor オブジェクト(の参照？) を `init_kernel` と `init_bias` に渡す(参照を渡してる？)
* `sess.run()` 実行時に `feed_dict` でそれぞれの値(`original_w` と `original_b`)を渡す

<details><summary>サンプルコード</summary><div>

```Python
# Get a handle on the assignment nodes for the hidden1 variables
graph = tf.get_default_graph()
assign_kernel = graph.get_operation_by_name("hidden1/kernel/Assign")
assign_bias = graph.get_operation_by_name("hidden1/bias/Assign")
init_kernel = assign_kernel.inputs[1]
init_bias = assign_bias.inputs[1]

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init, feed_dict={init_kernel: original_w, init_bias: original_b})
    # [...] 学習開始
    # 以下は確認用サンプル
    print(hidden1.eval(feed_dict={X: [[10.0, 11.0]]}))  # not shown in the book
    # => [[  61.   83.  105.]]
```

</div></details>

## 2.2.2 (比較的)冗長だが、明示的な方法

個人的にはこちらの方が明示してて分かりやすいですが、`variable` の定義や `placeholder` で型やサイズ(shape)を与える必要があるので冗長でやや効率が悪いとのことです(実行のパフォーマンスについてはわかってません…)。

おおまかな流れは以下の通りです。

* `variable` を定義
* 実行時の引数として `placeholder` を定義
* `tf.assign` で `varieble` に `placeholder` を割り当て
* `sess.run()` 実行時に `feed_dict` でそれぞれの値 (`original_w` と `original_b`) を渡す


<details><summary>サンプルコード</summary><div>

```Python
# Get a handle on the variables of layer hidden1
with tf.variable_scope("", default_name="", reuse=True):  # root scope
    hidden1_weights = tf.get_variable("hidden1/kernel")
    hidden1_biases = tf.get_variable("hidden1/bias")

# Create dedicated placeholders and assignment nodes
original_weights = tf.placeholder(tf.float32, shape=(n_inputs, n_hidden1))
original_biases = tf.placeholder(tf.float32, shape=n_hidden1)
assign_hidden1_weights = tf.assign(hidden1_weights, original_weights)
assign_hidden1_biases = tf.assign(hidden1_biases, original_biases)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    sess.run(assign_hidden1_weights, feed_dict={original_weights: original_w})
    sess.run(assign_hidden1_biases, feed_dict={original_biases: original_b})
    # [...] Train the model on your new task
    print(hidden1.eval(feed_dict={X: [[10.0, 11.0]]}))
```

</div></details>

## 2.3 Freezing the Lower Layers

ここまでで事前に学習した層を再利用できるようになりましたが、
上記の方法だと再利用した層のパラメータ(重みやバイアス等)も更新されます。
ここでは再利用した低層のパラメータを固定したまま学習させてみます。

固定させることで以下のようなメリットがあります。

* 低層のパラメータを更新する必要がないので学習が速くなる
* 後何か（あれば

また、ある層から下のパラメータを固定する方法は以下の2通りあります。
まだ使い分けるシチュエーションを知らないのですが、後者の方がモデル定義の時点で分かるので他の人が読んだときに意図が伝わりやすい気がします。


* `optimizer.minimize()` の引数 `var_list` で学習する変数を指定
* ネットワークの定義時に `tf.stop_gradient()` による層を挿入する

また事前に以下の共通部分は実行してあるとします。
今回は入力層、出力層、隠れ層4つからなるネットワークを構築します。
以下のように層によってパラメータの扱いが違うことに注意してください。

* 隠れ層 1,2 は他のモデルのパラメータを再利用し、かつパラメータを **更新しない**
* 隠れ層 3 は他のモデルのパラメータを再利用し、かつパラメータを **更新する**
* 隠れ層 4 と出力層は新しくパラメータを学習する

<details><summary>共通部分(各サンプルコードの前にそれぞれ実行する)</summary><div>

```Python
reset_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300 # 他のモデルのパラメータを再利用する。パラメータは更新しない
n_hidden2 = 50  # 他のモデルのパラメータを再利用する。パラメータは更新しない
n_hidden3 = 50  # 他のモデルのパラメータを再利用する。パラメータは "更新する"
n_hidden4 = 20  # 新しく学習する
n_outputs = 10  # 新しく学習する

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")
```

</div></details>


### 2.3.1 `optimizer.minimize()` の引数 `var_list` で学習する変数を指定

ポイントは以下2点

* `tf.get_collection()` で学習(更新)対象となる層のパラメータ名一覧を取得
* `optimizer.minimize()` の引数 `var_list` で上記のパラメータを指定

<details><summary>サンプルコード(事前に一度共通部分を実行する)</summary><div>

```Python
with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")       # reused
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2") # reused
    hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name="hidden3") # reused
    hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu, name="hidden4") # new!
    logits = tf.layers.dense(hidden4, n_outputs, name="outputs")                         # new!

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # ここで学習対象の変数名一覧を取得
    # scope は正規表現で指定可能
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   scope="hidden[34]|outputs")
    # ここで更新対象パラメータを指定
    training_op = optimizer.minimize(loss, var_list=train_vars)

# 実行前初期化
init = tf.global_variables_initializer()
new_saver = tf.train.Saver()

# 隠れ層1 から 3 のパラメータを再利用するために事前定義
# まだチェックポイントを特定していないことに注意
reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                               scope="hidden[123]") # regular expression
reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars])
restore_saver = tf.train.Saver(reuse_vars_dict) # to restore layers 1-3

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()
    # どこかで学習させてチェックポイント './my_model_final.ckpt' に保存済だとする
    # 隠れ層1から3のパラメータを指定
    restore_saver.restore(sess, "./my_model_final.ckpt")

    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images,
                                                y: mnist.test.labels})
        print(epoch, "Test accuracy:", accuracy_val)

    save_path = saver.save(sess, "./my_new_model_final.ckpt")
```

</div></details>


### 2.3.2 ネットワークの定義時に `tf.stop_gradient()` による層を挿入する

* `tf.stop_gradient(hidden2)` により勾配計算を途中でやめるための層が追加されている

<details><summary>サンプルコード(事前に共通部分を一度実行する)</summary><div>

```Python
with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu,
                              name="hidden1") # reused frozen
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu,
                              name="hidden2") # reused frozen
    hidden2_stop = tf.stop_gradient(hidden2)  # ここから上の勾配計算をやめる(≒ パラメータ更新をしない？)
    hidden3 = tf.layers.dense(hidden2_stop, n_hidden3, activation=tf.nn.relu,
                              name="hidden3") # reused, not frozen
    hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu,
                              name="hidden4") # new!
    logits = tf.layers.dense(hidden4, n_outputs, name="outputs") # new!

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

# 一つ前の手法と違って更新対象のパラメータを指定していない
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

# 再利用するための事前定義
reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                               scope="hidden[123]") # regular expression
reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars])
restore_saver = tf.train.Saver(reuse_vars_dict) # to restore layers 1-3

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "./my_model_final.ckpt")

    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images,
                                                y: mnist.test.labels})
        print(epoch, "Test accuracy:", accuracy_val)

    save_path = saver.save(sess, "./my_new_model_final.ckpt")
```

</div></details>


## 2.4 Caching the Frozen Layers

* Frozen Layers(固定層)のパラメータが変わらない
  * => 訓練データに対する一番上の Frozen Layer の出力も変わらない
* つまり出力結果をキャッシュとして保持しておけば、epoch の度にトレーニングデータの入力からせずに済む
  * (そこは TensorFlow で自動でやってくれない)
* 訓練データが大きかったり、epoch 数が多い場合に有効
* 一番上の Frozen Layer の出力をトレーニングデータとして予め変換して、その後 Frozen Layer ではない層を学習する感じ？

まずはモデルと再利用するための変数の事前定義を行います。
これは上記の `tf.stop_gradient()` を用いたネットワークと全く同じです。

隠れ層2までがパラメータ固定であることに注意してください。

<details><summary>サンプルコード(ネットワークと再利用するための変数定義)</summary><div>

```Python
reset_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300 # reused
n_hidden2 = 50  # reused
n_hidden3 = 50  # reused
n_hidden4 = 20  # new!
n_outputs = 10  # new!

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu,
                              name="hidden1") # reused frozen
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu,
                              name="hidden2") # reused frozen & cached
    hidden2_stop = tf.stop_gradient(hidden2)
    hidden3 = tf.layers.dense(hidden2_stop, n_hidden3, activation=tf.nn.relu,
                              name="hidden3") # reused, not frozen
    hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.relu,
                              name="hidden4") # new!
    logits = tf.layers.dense(hidden4, n_outputs, name="outputs") # new!

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

# 再利用するための変数定義
reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                               scope="hidden[123]") # regular expression
reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars])
restore_saver = tf.train.Saver(reuse_vars_dict) # to restore layers 1-3

init = tf.global_variables_initializer()
saver = tf.train.Saver()
```

</div></details>


ここからがキャッシュを行う部分です。
ポイントは以下の通りです。

* 事前(epoch 回す前)に `h2_cache` と `h2_cache_test` を計算する
  * サンプルコードではデータ(`mnist.train.images`、`mnist.test.images`)を一度に全て入力していますが、メモリが足りない場合は小分けに求める必要があります
* epoch 内では `X`, `y` の代わりに `h2_cache` と `hs_cache_test` を使う


<details><summary>サンプルコード(cache を利用した学習)</summary><div>

```Python
import numpy as np

n_batches = mnist.train.num_examples // batch_size

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "./my_model_final.ckpt")

    h2_cache = sess.run(hidden2, feed_dict={X: mnist.train.images})
    h2_cache_test = sess.run(hidden2, feed_dict={X: mnist.test.images}) # not shown in the book

    for epoch in range(n_epochs):
        # 0 から mnsits.train.num_examples - 1 までの数をランダムに並べ替えたもの
        shuffled_idx = np.random.permutation(mnist.train.num_examples)
        # ランダムに並べ替えて、バッチサイズで分割
        hidden2_batches = np.array_split(h2_cache[shuffled_idx], n_batches)
        y_batches = np.array_split(mnist.train.labels[shuffled_idx], n_batches)
        for hidden2_batch, y_batch in zip(hidden2_batches, y_batches):
            sess.run(training_op, feed_dict={hidden2:hidden2_batch, y:y_batch})

        accuracy_val = accuracy.eval(feed_dict={hidden2: h2_cache_test, # not shown
                                                y: mnist.test.labels})  # not shown
        print(epoch, "Test accuracy:", accuracy_val)                    # not shown

    save_path = saver.save(sess, "./my_new_model_final.ckpt")
```

</div></details>


## 2.5 Twaeking, Dropping, or Replacing the Upper Layers

* 学習済みモデルを適用(転移学習)するときに、上の層をどうするべきか
	* タスクや出力の次元が違うので一番上は変わる
	* 特に知見がない場合は、上から少しずつ変えていく
		* 元のモデルのパラメータのまま、可変(Unfrozen)にする -> だめならその層のパラメータを初期化する
		* その間下の層は元のモデルのままパラメータ固定(Frozen)

## 2.6 Model Zoos

* 画像認識用の pretrained model
* 予め学習してあるモデルやデータセットが利用できる
  * TensorFlow: https://github.com/tensorflow/models
  * Caffe: https://github.com/BVLC/caffe/wiki/Model-Zoo

## 2.7 Unsupervised Pretraining

* 一部のデータにしかラベルが付いておらず、ラベルがついていないデータが大量にある場合
* 事前学習として、教師なし学習を行わせてみると効果がある
  * Autoencoder
  * Restricted Boltzmann Machine(RBM)
* 学習させる際は、一層ずつ行い、一番上の層以外はパラメータを固定させる(Frozen)
  * 例
    * 入力層 + 隠れ層1(可変) で教師なし学習
    * 入力層 + 隠れ層1(固定) + 隠れ層2(可変) で教師なし学習
    * 入力層 + 隠れ層1(固定) + 隠れ層2(固定) + 隠れ層3(可変)で教師なし学習
    * 入力層 + 隠れ層1(固定) + 隠れ層2(固定) + 隠れ層3(固定) + 出力層(可変)で **教師あり学習** (本来やりたいタスク)

<!--
できれば図を用意したい
-->

## 2.8 Pretraining on an Auxiliary Task

* 補助タスクによる事前学習
  * ここで学習した低層を使いまわす
* 強化学習でよく使われているみたい？
  * 参考リンク: https://github.com/arXivTimes/arXivTimes/issues/56
* 例(画像による顔認証を行いたい場合)
  * やりたいこと: 手元のデータについて上手くクラス分類する
  * 各人について画像データが数枚しかないとする
    * 良い分類器を作るにはデータが十分でない
  * ここでネット上からランダムに顔画像を集め、
    * ２つの画像が同じ人の特徴か否か、といったタスクについて学習させる
  * ここで得た低層パラメータを使って本来やりたいことについて学習させる

# 3. Faster Optimization

* [Optimizer : 深層学習における勾配法について - Qiita](https://qiita.com/tokkuman/items/1944c00415d129ca0ee9)
* [勾配降下法の最適化アルゴリズムを概観する | POSTD](https://postd.cc/optimizing-gradient-descent/)
* 各オプティマイザ概要は上記の素晴らしい記事にまとめられているので、省略します。

TensorFlow でのコードは以下の通りです。

<details><summary>サンプルコード(各オプティマイザ指定)</summary><div>

```Python
# Momentum
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                       momentum=0.9)
# Nesterov Accelerated Gradient
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                       momentum=0.9, use_nesterov=True)
# AdaGrad
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
# RMSProp
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                      momentum=0.9, decay=0.9, epsilon=1e-10)
# Adam Optimization
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
```

</div></details>


## 3.1 Learning Rate Scheduling

学習率は他の機械学習アルゴリズムでも頻出パラメータですので、馴染みのある方が多いと思います。
ここでは `tf.exponential_decay()` を用いた学習率の調整方法を紹介します。
`tf.exponential_decay()` では以下のようにして学習率を更新していきます。


```math
\mathrm{decayed\_learning\_rate} = \mathrm{learning\_rate} * \mathrm{decay\_rate} ^ {\left(\frac{\mathrm{global\_step}}{\mathrm{decay\_steps}}\right)}
```
またコードについては、学習率調整箇所以外は上記の手法と変わりません。


<details><summary>サンプルコード(モデル定義)</summary><div>

```Python
reset_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_hidden2 = 50
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
```

</div></details>

<details><summary>サンプルコード(学習率調整箇所)</summary><div>

```Python
with tf.name_scope("train"):       # not shown in the book
    initial_learning_rate = 0.1
    decay_steps = 10000
    decay_rate = 1/10
    global_step = tf.Variable(0, trainable=False, name="global_step")
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,
                                               decay_steps, decay_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    training_op = optimizer.minimize(loss, global_step=global_step)
```

</div></details>


<details><summary>サンプルコード(学習と評価)</summary><div>

```Python
init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 5
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images,
                                                y: mnist.test.labels})
        print(epoch, "Test accuracy:", accuracy_val)

    save_path = saver.save(sess, "./my_model_final.ckpt")
```

</div></details>

# 4. Avoiding Overfitting Through Regularization

* ここでは過学習を抑制する方法について紹介します

## 4.1 Early Stopping

* その名の通り早めに学習を止めること。
* 実装する場合は以下のような方法がある
	* 一定ステップ毎にモデルを評価(ex. 50ステップ)
	* 評価がよかったときのモデル(のパラメータ)を保存
		* 次に評価したモデルの方がよかったら更新

## 4.2 L1 or L2 Regularization

* いずれも自分で定義するか TensorFlow 付属の関数で呼び出せる
* 損失を定義するところに正則化項を加える

### 4.2.1 自分で目的関数を定義

ここでは簡単のため隠れ層が一つのみの場合を考えます。
ポイントは `reg_losses = tf.reduce_sum(tf.abs(W1)) + tf.reduce_sum(tf.abs(W2))` です。
ここで正則化項を定義して元の損失関数に足しています。

またその際に `scale` を正則化項にかけています。これは正則化項の重みです。

<details><summary>サンプルコード</summary><div>

```Python
reset_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
    logits = tf.layers.dense(hidden1, n_outputs, name="outputs")

W1 = tf.get_default_graph().get_tensor_by_name("hidden1/kernel:0")
W2 = tf.get_default_graph().get_tensor_by_name("outputs/kernel:0")

scale = 0.001 # l1 regularization hyperparameter

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logits)
    base_loss = tf.reduce_mean(xentropy, name="avg_xentropy")
    reg_losses = tf.reduce_sum(tf.abs(W1)) + tf.reduce_sum(tf.abs(W2))
    # L2 正則化をしたい場合は下記の式を用いる
    # reg_losses = tf.reduce_sum(tf.squere(W1)) + tf.reduce_sum(tf.squere(W2))
    loss = tf.add(base_loss, scale * reg_losses, name="loss")

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 20
batch_size = 200

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images,
                                                y: mnist.test.labels})
        print(epoch, "Test accuracy:", accuracy_val)

    save_path = saver.save(sess, "./my_model_final.ckpt")
```

</div></details>


### 4.2.2 TensorFlow で用意されているものを使う

* `tf.layers.dense()` の 引数 `kernel_regularizer` で正則化方法を指定する
* また共通した引数を持つので `functools.partial()` で同じ引数を省略できる

<details><summary>サンプルコード</summary><div>

```Python
from functools import partial
reset_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_hidden2 = 50
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

scale = 0.001

my_dense_layer = partial(
    tf.layers.dense, activation=tf.nn.relu,
    kernel_regularizer=tf.contrib.layers.l1_regularizer(scale))  # ここで L1 正則化指定

with tf.name_scope("dnn"):
    # partial で同じ引数を省略
    hidden1 = my_dense_layer(X, n_hidden1, name="hidden1")
    hidden2 = my_dense_layer(hidden1, n_hidden2, name="hidden2")
    logits = my_dense_layer(hidden2, n_outputs, activation=None,
                            name="outputs")

with tf.name_scope("loss"):                                     # not shown in the book
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(  # not shown
        labels=y, logits=logits)                                # not shown
    base_loss = tf.reduce_mean(xentropy, name="avg_xentropy")   # not shown
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([base_loss] + reg_losses, name="loss")

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 20
batch_size = 200

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images,
                                                y: mnist.test.labels})
        print(epoch, "Test accuracy:", accuracy_val)

    save_path = saver.save(sess, "./my_model_final.ckpt")
```

</div></details>


## 4.3 DropOut

* 提案した論文: [[G. E. Hinton, 2012]](https://arxiv.org/pdf/1207.0580.pdf)
* 上記のもっと詳細: [[Nitish Srivastava, 2014]](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
* 各データの学習時に、ある層と層の間の全てのネットワークを使うのではなく、一部を除いて(Dropout した)学習する。
  * 各データ毎にネットワークが少し変わる
* アンサンブル学習となり、精度が上がるとのことです。
* `tf.layers.dropout()` をドロップさせる箇所に挿入する
  * `dropout_rate` はドロップ率
* 元の正解率が 95% でも 1-2% 上げちゃうくらい優秀な方法
* 入力のわずかな違いに過敏にならなくなる
* 入力層と隠れ層に対して適用される(出力層には適用しません)
* 小分けにしたネットワークによるアンサンブル学習となっている

<details><summary>サンプルコード</summary><div>

```Python
reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

training = tf.placeholder_with_default(False, shape=(), name='training')

dropout_rate = 0.5  # == 1 - keep_prob
X_drop = tf.layers.dropout(X, dropout_rate, training=training)

# Dropout する箇所を指定する
with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X_drop, n_hidden1, activation=tf.nn.relu,
                              name="hidden1")
    hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=training)
    hidden2 = tf.layers.dense(hidden1_drop, n_hidden2, activation=tf.nn.relu,
                              name="hidden2")
    hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=training)
    logits = tf.layers.dense(hidden2_drop, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 20
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={training: True, X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print(epoch, "Test accuracy:", acc_test)

    save_path = saver.save(sess, "./my_model_final.ckpt")
```

</div></details>



## 4.4 Max-Norm Regularization

* 各層の重み $\boldsymbol{w}$ に対して、その L2 ノルムが閾値 $r$ を超えないように正則化する。
* $\boldsymbol{w}$ の更新後に、もしその L2 ノルムが閾値を超えていたら下記のように更新する
	* $ {|| \cdot ||_2} $ は L2 ノルムを表す

```math
\boldsymbol{w} \leftarrow \boldsymbol{w}\frac{r}{||\boldsymbol{w}||_2} \space (\mathrm{if}  \space ||\boldsymbol{w}||_2 > r)
```


### 4.4.1 とりあえず実装

ポイントは以下2点

* `tf.clip_by_norm()` を使う。
*  毎 epoch 毎に正則化を実行する

<details><summary>サンプルコード(モデル定義)</summary><div>

```Python
reset_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 50
n_outputs = 10

learning_rate = 0.01
momentum = 0.9

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    training_op = optimizer.minimize(loss)    

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
```

</div></details>


<details><summary>サンプルコード(Max-Norm Regularization 実装部分)</summary><div>

```Python
threshold = 1.0
weights = tf.get_default_graph().get_tensor_by_name("hidden1/kernel:0")
clipped_weights = tf.clip_by_norm(weights, clip_norm=threshold, axes=1)
clip_weights = tf.assign(weights, clipped_weights)

weights2 = tf.get_default_graph().get_tensor_by_name("hidden2/kernel:0")
clipped_weights2 = tf.clip_by_norm(weights2, clip_norm=threshold, axes=1)
clip_weights2 = tf.assign(weights2, clipped_weights2)
```

</div></details>


<details><summary>サンプルコード(学習と評価部分)</summary><div>

```Python
init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 20
batch_size = 50

with tf.Session() as sess:                                             
    init.run()                                                          
    for epoch in range(n_epochs):                                       
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)       
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            clip_weights.eval()  # 実行
            clip_weights2.eval()  # 実行                                   
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images,       
                                            y: mnist.test.labels})      
        print(epoch, "Test accuracy:", acc_test)                        

    save_path = saver.save(sess, "./my_model_final.ckpt")               
```

</div></details>


### 4.4.2 関数を定義して少しスマートに実装

上記のコードを $l_1$ or $l_2$ Regularization のコードと比べると、なんだかスマートじゃない気がしてきます。ここでは自身で関数を定義して上手くモデルの中に入るようにします。

<details><summary>サンプルコード(正則化用関数定義)</summary><div>

```Python
def max_norm_regularizer(threshold, axes=1, name="max_norm", collection="max_norm"):
    def max_norm(weights):
        clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
        clip_weights = tf.assign(weights, clipped, name=name)
        tf.add_to_collection(collection, clip_weights)  # ここで名前定義
        return None # there is no regularization loss term
		# 関数を返している
    return max_norm
```

</div></details>



<details><summary>サンプルコード(モデル定義)</summary><div>

```Python
reset_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 50
n_outputs = 10

learning_rate = 0.01
momentum = 0.9

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

max_norm_reg = max_norm_regularizer(threshold=1.0)

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu,
                              kernel_regularizer=max_norm_reg, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu,
                              kernel_regularizer=max_norm_reg, name="hidden2")
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    training_op = optimizer.minimize(loss)    

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
```

</div></details>


コードにある通り、うまく `tf.layers.dense()` の引数として入れることができてます。

<details><summary>サンプルコード(学習と評価)</summary><div>

```Python
init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 20
batch_size = 50

clip_all_weights = tf.get_collection("max_norm")

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            sess.run(clip_all_weights)
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images,
                                            y: mnist.test.labels})
        print(epoch, "Test accuracy:", acc_test)                  

    save_path = saver.save(sess, "./my_model_final.ckpt")         
```

</div></details>


## 4.5 Data Augmentation

* もとのデータを少し加工したデータを作ることでデータの量を増やすこと。
* 例えば画像の場合以下のような加工ができる(これらをまとめて行列で変換するのがアフィン変換)
	* 回転
	* サイズ変更
	* 反転
	* 平行移動

# 5. Pratical Guidelines

とりあえず事前知識が何もない場合は下記の構成を一旦試してみると良いと思います。

## とりあえず試してみるための構成

|項目|手法|
|:-------------------------|:---------------------------------|
|Initialization(初期化方法) | He initialization |
|Activation function(活性化関数) | ELU |
|Normalization(正規化)| Batch Normalization |
|Regularization| Dropout |
|Optimizer| Nesterov Accelerated Gradient(NAG) ※1 |
|Learning rate schedule | None |

* ※1: NAG は `tf.train.MomentumOptimizer()` の引数で `use_nesterov=True` を指定すると使えます
  * ただ Qiita とかだと Adam 多いし、そもそも最適化手法はたくさんあるので時間が許すなら色々試した方が良いと思います
  * 参考リンク: [深層学習の最適化アルゴリズム - Qiita](https://qiita.com/ZoneTsuyoshi/items/8ef6fa1e154d176e25b8#adasecant-2017)

またケースによっては下記のような微調整が必要になるかもしれません

## 微調整項目

| ケース | 対処法 |
|:----------------------------|:-----------------|
|収束が遅すぎる、早すぎる | 学習率を上げ下げしたり、exponential decay を試してみる |
|学習データが少ない | data augmentation してみる |
| スパースなモデルが必要 | $l_1$ 正則化を加える。小さい weight を 0 にする |
| 実行時間(評価時間) を短くしたい | Batch Normalization やめる、ReLU 使う |


# 最後に

TensorFlow は僕には早すぎました

# Exercises

モチベがあれば別記事にて
