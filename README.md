# Weight Normalization

This repository includes an implemetantion of *Weight Normalization* for [Chainer](https://github.com/pfnet/chainer).
Weight normalization can help optimization of a model.

See [Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks](https://arxiv.org/pdf/1602.07868.pdf), Tim Salimans, Diederik P. Kingma, NIPS 2016


## How To Use

If you just wrap a link with my function `convert_with_weight_normalization`,
the link is transformed to one using weight normalization.

For example, when you define a link with weight normalization,
you can write
```
l1 = WN.convert_with_weight_normalization(
    L.Linear, 784, 100)
c1 = WN.convert_with_weight_normalization(
    L.Convolution2D, 1, 128, ksize=3, pad=1, wscale=2.)
```
instead of
```
l1 = L.Linear(784, 100)
c1 = L.Convolution2D(1, 128, ksize=3, pad=1, wscale=2.)
```
which is a common writing respectively.


Note: 
- This function is tested only for `chainer.links.Linear`, `chainer.links.Convolution2D` and `chainer.links.LSTM`. Thus, this can not guarantee that this will work for other untested links which have parameters of `ndim >= 2` (e.g., `chainer.links.ConvolutionND`, `chainer.links.Deconvolution2D`).


## Experiment

This is an experiment to train 10-layer MLP on MNIST using SGD with a bad learning rate (lr=1.0).
Code is derived and modified from Chainer official example.

Result with weight normalization:
```
# unit: 128
# Minibatch-size: 100
# epoch: 20

epoch       main/loss   validation/main/loss  main/accuracy  validation/main/accuracy  elapsed_time
1           2.06877     1.52957               0.18655        0.2723                    16.9592
2           1.04963     0.609608              0.582749       0.7827                    35.3212
3           0.683584    0.483358              0.765717       0.8992                    53.0184
4           0.320896    0.295775              0.915051       0.9182                    69.6802
5           0.20813     0.162944              0.945167       0.9586                    85.7245
6           0.152095    0.163007              0.959518       0.9559                    101.133
7           0.126859    0.141788              0.965866       0.9629                    116.564
8           0.108074    0.120223              0.970883       0.9688                    131.995
9           0.0962295   0.117946              0.973883       0.9679                    147.381
10          0.0851106   0.123432              0.977049       0.9684                    162.969
11          0.0748157   0.116313              0.979482       0.9699                    179.855
12          0.0693717   0.104516              0.980316       0.9731                    195.235
13          0.0602229   0.115475              0.982882       0.9703                    212.155
14          0.0552534   0.117155              0.984349       0.9698                    228.243
15          0.0511129   0.110413              0.985666       0.9718                    244.462
16          0.0484171   0.109682              0.985865       0.9726                    260.538
17          0.0442805   0.113194              0.987532       0.9721                    277.652
18          0.0407127   0.107786              0.988315       0.974                     294.323
19          0.0376482   0.11264               0.989715       0.973                     310.307
20          0.0365301   0.141542              0.989698       0.9688                    325.875
```

Result without weight normalization. Training failed.:
```
# unit: 128
# Minibatch-size: 100
# epoch: 20

epoch       main/loss   validation/main/loss  main/accuracy  validation/main/accuracy  elapsed_time
1           2.31399     2.30319               0.11065        0.0974                    8.23641
2           2.30359     2.30396               0.1072         0.1135                    16.1239
3           2.30352     2.30443               0.107633       0.101                     23.605
4           2.30354     2.30405               0.106217       0.1032                    30.4258
5           2.30359     2.30463               0.10735        0.1135                    38.2871
6           2.30326     2.30494               0.109617       0.1135                    46.1897
7           2.30364     2.30237               0.106233       0.1135                    53.0346
8           2.30348     2.30281               0.106183       0.098                     58.9748
9           2.30356     2.3032                0.106533       0.0974                    64.8374
10          2.30344     2.30257               0.1076         0.1135                    71.1814
11          2.30365     2.30289               0.10545        0.1135                    77.394
12          2.3037      2.30301               0.10685        0.1135                    83.2978
13          2.30365     2.30259               0.10735        0.1032                    89.3203
14          2.30369     2.30131               0.107017       0.1135                    95.4914
15          2.3037      2.30292               0.104633       0.1028                    101.821
16          2.30389     2.30172               0.107183       0.1135                    108.227
17          2.30375     2.30178               0.107083       0.1028                    114.334
18          2.30359     2.30404               0.106317       0.1135                    120.461
19          2.30371     2.30483               0.107433       0.098                     126.327
20          2.30359     2.30305               0.105483       0.1135                    132.842
```
