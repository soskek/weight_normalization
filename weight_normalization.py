import chainer
from chainer import cuda
from chainer import functions as F


def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(x.shape[0], -1)


def get_norm(W, expand=False):
    xp = cuda.get_array_module(W)
    norm = xp.linalg.norm(_as_mat(W), axis=1) + 1e-12
    for i in range(W.ndim):
        if norm.ndim <= i:
            norm = xp.expand_dims(norm, axis=i)
    return norm


def normalize(W):
    norm = get_norm(W, expand=True)
    return W / norm


def get_norm_variable(W, expand=False):
    norm = F.sqrt(F.sum(_as_mat(W) ** 2, axis=1) + 1e-12)
    if expand:
        for i in range(W.ndim):
            if norm.ndim <= i:
                norm = F.expand_dims(norm, axis=i)
    return norm


def normalize_variable(W):
    norm = get_norm_variable(W, expand=True)
    return W / F.broadcast_to(norm, W.shape)


def convert_with_weight_normalization(link_class, *args1, **args2):
    """Weight Normalization Transformer

    This function transforms a link to a variant using weight normalization
    by decomposing a link's parameter `W` into
    a direction component `W_v` and a norm component `W_g`
    without large changes of interface.
    Lazy dimension setup of a parameter (e.g., `L.Linear(None, 128)`)
    is currently not supported.

    Note: This function is tested only for :class:`~chainer.links.Linear`
    and :class:`~chainer.links.Convolution2D`.
    Thus, this can not guarantee that this will work for
    other untested links which have a `W` parameter
    (e.g., :class:`~chainer.links.ConvolutionND`,
    :class:`~chainer.links.Deconvolution2D`).

    TODO: add initialization technieque for weight normalization

    See: https://arxiv.org/pdf/1602.07868.pdf

    Args:
        link_class (:class:`~chainer.Link`):
            A Link class such as :class:`~chainer.links.Linear`.
        args (anything): Argument inputs for the given link class.

    Returns:
        An link object of the given link class using weight normalization.

    .. admonition:: Example

        >>> x = np.array([[-1, 0], [2, -3]], 'f')
        >>> wn_l = convert_with_weight_normalization(L.Linear, 2, 5)
        >>> y = wn_l(x)

    """

    class WeightNormalizedLink(link_class):

        def __init__(self, *_args1, **_args2):
            super(WeightNormalizedLink, self).__init__(*_args1, **_args2)
            W = getattr(self, 'W')
            assert(isinstance(W, chainer.Variable))
            delattr(self, 'W')
            self._params.remove('W')
            self.add_param('W_v', W.shape)
            getattr(self, 'W_v').data[:] = normalize(W.data)
            self.add_param('W_g', (W.shape[0], ) + (1, ) * (W.ndim - 1))
            getattr(self, 'W_g').data[:] = \
                get_norm(W.data, expand=True)
            assert(self.xp.all(
                abs(W.data - (F.broadcast_to(self.W_g, self.W_v.shape) *
                              normalize_variable(self.W_v)).data) < 1e-4))
            setattr(self, '_after_setup', True)

        def __getattribute__(self, name):
            if name == 'W' and getattr(self, '_after_setup', False):
                return F.broadcast_to(self.W_g, self.W_v.shape) * \
                    normalize_variable(self.W_v)
            else:
                return object.__getattribute__(self, name)

    return WeightNormalizedLink(*args1, **args2)


if __name__ == '__main__':
    from chainer import links as L
    from chainer import testing
    import numpy

    n_in, n_out = 3, 5
    l = convert_with_weight_normalization(L.Linear, n_in, n_out)
    assert(l.W.creator is not None)
    assert(l.W_g.creator is None)
    assert(l.W_v.creator is None)
    testing.assert_allclose(
        l.W_g.data * F.normalize(l.W_v, axis=1).data, l.W.data,
        rtol=1e-5)
    testing.assert_allclose(
        l.W_g.data * l.W_v.data, l.W.data,
        rtol=1e-5)
    W, W_g, W_v = l.W.data + 0, l.W_g.data + 0, l.W_v.data + 0
    opt = chainer.optimizers.SGD()
    opt.setup(l)
    l.cleargrads()
    loss = F.sum(l(numpy.random.rand(10, 3).astype('f')) ** 2)
    loss.backward()
    opt.update()
    assert(numpy.all(W != l.W.data))
    assert(numpy.all(W_g != l.W_g.data))
    assert(numpy.all(W_v != l.W_v.data))
    testing.assert_allclose(
        l.W_g.data * F.normalize(l.W_v, axis=1).data, l.W.data,
        rtol=1e-5)

    n_in, n_out, ksize = 2, 4, 3
    l = convert_with_weight_normalization(
        L.Convolution2D, n_in, n_out, ksize=ksize, pad=1, wscale=2.)
    assert(l.W.creator is not None)
    assert(l.W_g.creator is None)
    assert(l.W_v.creator is None)
    normalized_W_v = F.normalize(l.W_v.reshape(
        (n_out, n_in * ksize * ksize)), axis=1).reshape(l.W_v.shape).data
    testing.assert_allclose(
        l.W_g.data * normalized_W_v, l.W.data,
        rtol=1e-5)
    testing.assert_allclose(
        l.W_g.data * l.W_v.data, l.W.data,
        rtol=1e-5)
    W, W_g, W_v = l.W.data + 0, l.W_g.data + 0, l.W_v.data + 0
    opt = chainer.optimizers.SGD()
    opt.setup(l)
    l.cleargrads()
    loss = F.sum(l(numpy.random.rand(10, n_in, 20, 20).astype('f')) ** 2)
    loss.backward()
    opt.update()
    assert(numpy.all(W != l.W.data))
    assert(numpy.all(W_g != l.W_g.data))
    assert(numpy.all(W_v != l.W_v.data))
    normalized_W_v = F.normalize(l.W_v.reshape(
        (n_out, n_in * ksize * ksize)), axis=1).reshape(l.W_v.shape).data
    testing.assert_allclose(
        l.W_g.data * normalized_W_v, l.W.data,
        rtol=1e-5)
