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
- This function only supports links without grand children paramters. For example, `chainer.links.LSTM` is not supported. (If you transform its internal `Linear` directly, it may work.)
- This function is tested only for `chainer.links.Linear` and `chainer.links.Convolution2D`. Thus, this can not guarantee that this will work for other untested links which have parameters of `ndim >= 2` (e.g., `chainer.links.ConvolutionND`, `chainer.links.Deconvolution2D`).


This function works in both chainer v1 and v2 (current).


## Experiment

This is an experiment to train 10-layer residual NN with activations of leaky-relu on MNIST using SGD with a bad learning rate (lr=0.1).
Code is derived and modified from Chainer official example.

Result with weight normalization:
```
iteration   main/loss   validation/main/loss  main/accuracy  validation/main/accuracy  elapsed_time
1           5.80999                           0.08                                     0.64273
2           10.8657                           0.14                                     0.679493
3           5.57372                           0.09                                     0.702814
4           2.66745                           0.22                                     0.722323
5           2.22639                           0.14                                     0.741015
6           2.36748                           0.2                                      0.758818
7           2.24198                           0.14                                     0.775544
8           2.08068                           0.3                                      0.792325
9           2.14593                           0.26                                     0.809057
10          1.93175                           0.44                                     0.825989
11          1.65397                           0.48                                     0.842731
12          1.63579                           0.49                                     0.859861
13          1.93049                           0.24                                     0.877223
14          2.59935                           0.09                                     0.894272
15          2.39911                           0.22                                     0.911247
16          1.83692                           0.51                                     0.928246
17          1.58412                           0.5                                      0.945195
18          1.59829                           0.48                                     0.962008
19          1.59301                           0.4                                      0.978866
20          1.48322                           0.48                                     0.99572
...
...
595         0.292018                          0.89                                     13.6053
596         0.106104                          0.94                                     13.6312
597         0.101768                          0.95                                     13.6571
598         0.209457                          0.95                                     13.6829
599         0.233117                          0.89                                     13.7088
600         0.237073    0.284317              0.91           0.9122                    14.6295
```

Result without weight normalization. Training failed due to explosion.
```
iteration   main/loss   validation/main/loss  main/accuracy  validation/main/accuracy  elapsed_time
1           5.80999                           0.08                                     0.753265
2           16.8105                           0.09                                     0.765204
3           16.916                            0.13                                     0.775952
4           10.3611                           0.1                                      0.786683
5           32.9404                           0.08                                     0.797532
6           36.7193                           0.17                                     0.808314
7           9288.8                            0.1                                      0.819087
8           1.1515e+19                        0.05                                     0.829958
9           nan                               0.08                                     0.841378
10          nan                               0.11                                     0.852247
11          nan                               0.08                                     0.863049
12          nan                               0.04                                     0.874359
13          nan                               0.1                                      0.885276
14          nan                               0.06                                     0.896156
15          nan                               0.1                                      0.907041
16          nan                               0.14                                     0.918859
17          nan                               0.06                                     0.929799
18          nan                               0.07                                     0.940751
19          nan                               0.11                                     0.951689
20          nan                               0.13                                     0.962659
...
...
595         nan                               0.13                                     9.35053
596         nan                               0.13                                     9.36868
597         nan                               0.09                                     9.38676
598         nan                               0.11                                     9.4049
599         nan                               0.09                                     9.42308
600         nan         nan                   0.05           0.098                     9.97829
```
