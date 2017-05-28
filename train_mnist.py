#!/usr/bin/env python

from __future__ import print_function

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

import numpy

import weight_normalization as WN

# Network definition


class WNMLP(chainer.Chain):

    def __init__(self, n_units, n_out, n_layers=3):
        super(WNMLP, self).__init__(
            l1=WN.convert_with_weight_normalization(
                L.Linear, 784, n_units),
            lo=WN.convert_with_weight_normalization(
                L.Linear, n_units, n_out),
        )
        self.n_layers = n_layers
        for i in range(2, self.n_layers - 1):
            self.add_link(
                'l{}'.format(i),
                WN.convert_with_weight_normalization(
                    L.Linear, n_units, n_units))

    def __call__(self, x):
        act = F.relu
        h = x
        for i in range(1, self.n_layers - 1):
            h = act(getattr(self, 'l{}'.format(i))(h))
        return self.lo(h)


class MLP(chainer.Chain):

    def __init__(self, n_units, n_out, n_layers=3):
        super(MLP, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(None, n_units),
            lo=L.Linear(None, n_out),
        )
        self.n_layers = n_layers
        for i in range(2, self.n_layers - 1):
            self.add_link('l{}'.format(i), L.Linear(None, n_units))

    def __call__(self, x):
        act = F.relu
        h = x
        for i in range(1, self.n_layers - 1):
            h = act(getattr(self, 'l{}'.format(i))(h))
        return self.lo(h)


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=128,
                        help='Number of units')
    parser.add_argument('--use-wn', '-wn', action='store_const',
                        const=True, default=False)
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    numpy.random.seed(777)
    if args.use_wn:
        model = L.Classifier(WNMLP(args.unit, 10, n_layers=10))
    else:
        model = L.Classifier(MLP(args.unit, 10, n_layers=10))
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    # optimizer = chainer.optimizers.Adam()
    optimizer = chainer.optimizers.SGD(lr=1.)
    optimizer.setup(model)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()

    numpy.random.seed(777)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
