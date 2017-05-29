import chainer
from chainer import cuda
from chainer import functions as F
from chainer.utils import array


def get_norm(W, expand=False):
    xp = cuda.get_array_module(W)
    norm = xp.linalg.norm(array.as_mat(W), axis=1) + 1e-12
    if expand:
        expanded_shape = (W.shape[0], ) + (1, ) * (W.ndim - 1)
        norm = norm.reshape(expanded_shape)
    return norm


def normalize(W):
    norm = get_norm(W, expand=True)
    return W / norm


def get_norm_variable(W, expand=False):
    norm = F.sqrt(F.batch_l2_norm_squared(W) + 1e-12)
    if expand:
        expanded_shape = (W.shape[0], ) + (1, ) * (W.ndim - 1)
        norm = norm.reshape(expanded_shape)
    return norm


def normalize_variable(W):
    norm = get_norm_variable(W, expand=True)
    return W / F.broadcast_to(norm, W.shape)


def convert_with_weight_normalization(link_class, *args1, **args2):
    """Weight Normalization Transformer

    This function transforms a link to a variant using weight normalization
    by decomposing a link's each parameter `W` of `ndim >= 2` into
    a direction component `W_v` and a norm component `W_g`
    without large changes of interface.
    Lazy dimension setup of a parameter (e.g., `L.Linear(None, 128)`)
    is currently not supported.

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
            self._W_params = []
            _getattr = object.__getattribute__
            for name, param in list(self.namedparams()):
                if param.ndim < 2:
                    continue
                name = name.lstrip('/')
                W = param
                assert(isinstance(W, chainer.Variable))
                parent = self
                while '/' in name:
                    parent = _getattr(parent, name.split('/')[0])
                    name = name[name.index('/') + 1:]
                if not hasattr(parent, '_W_params'):
                    parent._W_params = []
                delattr(parent, name)
                parent._params.remove(name)
                parent._W_params.append(name)
                parent.add_param(name + '_v', W.shape)
                _getattr(parent, name + '_v').data[:] = normalize(W.data)
                parent.add_param(name + '_g',
                                 (W.shape[0], ) + (1, ) * (W.ndim - 1))
                _getattr(parent, name + '_g').data[:] = \
                    get_norm(W.data, expand=True)

        def __getattribute__(self, name):
            if name == '_W_params':
                return object.__getattribute__(self, name)

            if name in getattr(self, '_W_params', []):
                W_g = getattr(self, name + '_g')
                W_v = getattr(self, name + '_v')
                return F.broadcast_to(W_g, W_v.shape) * \
                    normalize_variable(W_v)
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

    n_in, n_out = 3, 5
    l = convert_with_weight_normalization(L.LSTM, n_in, n_out)
    for name, param in l.namedparams():
        if param.ndim < 2:
            continue
        name = name.lstrip('/')
        parent = l
        while '/' in name:
            parent = getattr(parent, name.split('/')[0])
            name = name[name.index('/') + 1:]
        if 'name' not in getattr(parent, '_W_params', []):
            continue
        _W = getattr(parent, name)
        _W_g = getattr(parent, name + '_g')
        _W_v = getattr(parent, name + '_v')
        assert(_W.creator is not None)
        assert(_W_g.creator is None)
        assert(_W_v.creator is None)
        testing.assert_allclose(
            _W_g.data * F.normalize(_W_v, axis=1).data, _W.data,
            rtol=1e-5)
        testing.assert_allclose(
            _W_g.data * _W_v.data, _W.data,
            rtol=1e-5)
        W, W_g, W_v = _W.data + 0, _W_g.data + 0, _W_v.data + 0
        opt = chainer.optimizers.SGD()
        opt.setup(l)
        l.cleargrads()
        loss = F.sum(l(numpy.random.rand(10, 3).astype('f')) ** 2)
        loss.backward()
        opt.update()

        _W = getattr(parent, name)
        _W_g = getattr(parent, name + '_g')
        _W_v = getattr(parent, name + '_v')
        assert(numpy.all(W != _W.data))
        assert(numpy.all(W_g != _W_g.data))
        assert(numpy.all(W_v != _W_v.data))
        testing.assert_allclose(
            _W_g.data * F.normalize(_W_v, axis=1).data, _W.data,
            rtol=1e-5)
