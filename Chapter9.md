<!--
$theme: gaia
prerender: true
page_number: true
$width: 12in
$size: 4:3
-->

### Hands on Machine Learning with
###   Scikit-Learn &TensolFlow
## Chapter 9
#### Up and Running with TensorFlow

###### Created by Yusuke FUJIMOTO

---

# はじめに

*  この資料は「[Hands-On Machine Learning with Scikit-Learn and TensorFlow - O'Reilly Media](http://shop.oreilly.com/product/0636920052289.do) 」
を読んだ際の（主にソースコードに関する）簡単な解説を残したものです。
*  全部を解説したわけではないので注意
*  余裕があればソースコード周りの背景知識もまとめたい
*  何かあったら yukkyo12221222@gmail.com まで

---

# Chapter 9
# Up and Running with TensorFlow

---

#### ポイント

* TensorFlow
   * google 製の計算ライブラリ（機械学習に限らない）
* 実行時は、主に以下の2ステップある
   * 計算グラフ（≒設計図）を作る
   * 計算を実行する
* テンソル（多次元配列）計算を扱う
   * 例：0次元テンソル＝スカラー、1次元テンソル＝ベクトル、2次元テンソル=行列、etc…


---
* なんで計算グラフ作るの？直接計算すれば良くない？
  * ニューラルネットワークの学習時は、微分が相互に影響し合う。それを手で全部書き下すのは厳しい
* 変数は Session 内でのみ有効
* 変数使用時は必ず初期化する

---

#### TF の関数メモ

```python
tf.variable()  # 変数定義
tf.constant()  # 定数定義
tf.transpose() # 転置
tf.matmul()    # 行列としてのかけざん
tf.random_uniform()  # 乱数生成, np.rand() 同じように使える
tf.assign()    # 値を書き換える（更新する）
tf.reduce_mean()  # 平均を求める（meanではないので注意）
```

---

#### Implementing Grandient Descent

* 勾配法を tensorflow で実装する。以下の2通りで。
   * 手で勾配を計算する
   * 自動で tensorflow に微分させる

```python
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
grad = 2/m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta,
                        theta - learning_rate * grad)
```

自動微分させる場合は以下のようにしてできる
`gradients = tf.gradients(mse, [theta])[0]`

---

| Technique             | 勾配を計算するのに グラフを横切る数？ | 精度 | その他              |
|-----------------------|---------------------------------------|------|---------------------|
| Numerical diff        | $n_{inputs} + 1$                      | Low  | 実装が楽            |
| Symbolic diff         | N/A                                   | High | 色んなグラフ作れる  |
| Forward-mode autodiff | $n_{inputs}$                          | High | Uses *dual numbers* |
| Reverse-mode autodiff | $n_{outputs} + 1$                     | High | TF で実装されてる   |

---

#### Using an Optimizer

勾配法以外にも学習（パラメータの更新）ができる

```python
optimizer = tf.train.GradientDescentOptimizer(
                       learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
```

---

#### Feeding Data to the Training Algorithm

* TFでは種類の数が扱える
  * Constant(定数)
  * Variable(変数)
  * Place holder
* 変数は Session に紐づく
* 変数を扱う際は初期化が必要
  * `sess.run(x.initializer)`
  * `init = tf.global_variables_initializer()`
    * session 内で `init.run()` を実行
---

* Placeholder
  * プレースホルダー（placeholder）はデータが格納される予定地
  * データは未定のままグラフを構築し、具体的な値は実行するときに与える
  * `feed_dict` で実行時に値を与える

```python
A = tf.placeholder(tf.float32, shape=(None, 3))  # 宣言
B = A + 5
with tf.Session() as sess:
    # 実行時に与える
    B_val_1 = B.eval(
        feed_dict={A: [[1, 2, 3]]})  # => [[6. 7. 8.]]
    B_val_2 = B.eval(
        feed_dict={A: [[4, 5, 6], [7, 8, 9]]}
        )  # => [[9.  10.  11.] [12. 13. 14.]]
```

---

#### Saving the Restoreing Models

* 参考リンク: [Qiita](https://qiita.com/yukiB/items/a7a92af4b27e0c4e6eb2)
* 注意事項
  * 保存したいすべての変数を宣言したあとで saver を宣言する

```python
saver = tf.train.Saver()
save_path = saver.save(sess, "/tmp/my_model.ckpt")
# これだけでも save される
saver.save(sess, "/tmp/my_model.ckpt")
# v1をmy_v1として保存
tf.train.Saver('my_v1': v1)
# v1, v2のみ保存
tf.train.Saver([v1, v2])
# 読み込み
saver.restore(sess, "/tmp/my_model_final.ckpt")
```

---

#### Visualizing the Graph and Training Curves Using Tensorboard

```python
# ログ用の初期設定
from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)
mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
# ログを取るとき（ループ中とか）
file_writer.add_summary(summary_str, step)
# 取り終えたら
file_writer.close()
```

見るときは bash 上で、`tensorboard --logdir tf_logs/`

---

#### Name Scopes

#### Modularity


#### Sharing Variables


### Exercises

* 省略。やらないとまずい

---

##### 参考サイト
* [TensorFlow入門 ― 変数とプレースホルダー - Build Insider](http://www.buildinsider.net/small/booktensorflow/0103)
* [TensorFlow学習パラメータのsave, restoreでつまった - Qiita](https://qiita.com/yukiB/items/a7a92af4b27e0c4e6eb2)
* aaa