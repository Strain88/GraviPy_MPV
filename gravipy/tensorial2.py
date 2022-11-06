"""
GraviPy.

Tensorial module implements a tensor components objects for
Coordinates, Metric, Christoffel, Ricci, Riemann etc.
"""

from collections import OrderedDict
import sympy
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

formatter = logging.Formatter(
    '%(module)s:%(name)s:%(funcName)s:%(lineno)d:%(message)s'
)

# file_handler = logging.FileHandler('sample.log')
# file_handler.setLevel(logging.DEBUG)
# file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

# logger.addHandler(file_handler)
logger.addHandler(stream_handler)


class GeneralTensor:
    """GeneralTensor.

    Represents tensor components objects in a particular Coordinate System.
    Tensor class should be extended rather than GeneralTensor class to create
    a new Tensor object.
    """

    GeneralTensorObjects = []

    def __new__(cls, *args, **kwargs):
        logger.debug(
            f'вызов __new__ для {cls} с параметрами {args} и {kwargs}\n'
        )
        return super().__new__(cls)

    @classmethod
    def update_tensor_object(cls, value):
        logger.debug(f'A number of tensors increased, the {value} is added')
        cls.GeneralTensorObjects.append(value)
        logger.debug(
            f'общее количество объектов = {len(cls.GeneralTensorObjects)}\n'
        )

    def __init__(
        self, symbol, rank, coords, metric=None, conn=None, *args, **kwargs
    ):
        self.update_tensor_object(self)   # changed 1
        self.is_tensor = True
        self.is_coordinate_tensor = False
        self.is_metric_tensor = False
        self.is_connection = False
        self.symbol = sympy.Symbol(symbol)
        self.rank = rank
        self.coords = coords
        self.metric = metric
        self.conn = conn
        self.dim = len(coords)
        self.components = {}
        self.partial_derivative_components = {}
        self.covariant_derivative_components = {}
        self.index_values = {
            -1: list(range(-self.dim, 0)),
            0: list(range(-self.dim, 0)) + list(range(1, self.dim + 1)),
            1: list(range(1, self.dim + 1)),
        }
        logger.debug(f'{self.index_values=}')
        self._set_index_types(kwargs)
        logger.debug(f'{self.index_types=}')
        if 'apply_tensor_symmetry' in kwargs:
            self.apply_tensor_symmetry = bool(kwargs['apply_tensor_symmetry'])
        else:
            self.apply_tensor_symmetry = False   # apply_tensor_symmetry
        logger.debug(f'{self.apply_tensor_symmetry=}\n')

    def __call__(self, *idxs):
        logger.debug(f'передаем индексы {idxs=}')
        current_mode = self._proper_tensor_indexes(idxs)
        logger.debug(f'{current_mode=}')
        if current_mode == 'component_mode':
            idxs = tuple(map(int, idxs))
            if idxs in self.components.keys():
                return self.components[idxs]
            else:
                if all([idxs[i] > 0 for i in range(self.rank)]):
                    return self._compute_covariant_component(idxs)
                else:
                    return self._compute_exttensor_component(idxs)
        elif current_mode == 'matrix_mode':
            return self._matrix_form(idxs)
        else:
            logger.error('Unexpected tensor index error')
            raise GraviPyError('Unexpected tensor index error')

    def _matrix_form(self, idxs):
        logger.debug(f'передаем индексы {idxs=}')
        allidxs = {
            i + 1: idxs[i] / abs(idxs[i])
            for i in range(self.rank)
            if isinstance(abs(idxs[i]), AllIdx)
        }
        logger.debug(f'что лежит в All {allidxs=}')
        if len(allidxs) % 2 != 0:
            allidxs.update({0: 0})
        allidxs = OrderedDict(sorted(allidxs.items(), key=lambda t: t[0]))
        logger.debug(f'All в упорядоченном {allidxs=}')
        paidxs = tuple(
            (
                list(allidxs.keys())[i : i + 2]
                for i in range(len(allidxs))
                if i % 2 == 0
            )
        )
        logger.debug(f'Пары индексов {paidxs=}')
        nidxs = list(idxs)
        logger.debug(f'передаем индексы {nidxs=}')
        logger.debug(f'{allidxs=}\n {paidxs=}\n {nidxs=}\n')

        def _rnidxs(idxs, pair, allidxs, k, l):
            logger.debug(f'{idxs=}, {pair=}, {allidxs=}, {k=}, {l=}')
            for i in range(len(idxs)):
                if i + 1 == pair[0]:
                    idxs[i] = k * allidxs[pair[0]]
                if i + 1 == pair[1]:
                    idxs[i] = l * allidxs[pair[1]]
            logger.debug(f'{idxs=}')
            return idxs

        if paidxs[0][0] == 0:
            M = sympy.Matrix(
                1,
                self.dim,
                lambda k, l: self.__call__(
                    *(_rnidxs(nidxs, paidxs[0], allidxs, k + 1, l + 1))
                ),
            )
        else:
            M = sympy.Matrix(
                self.dim,
                self.dim,
                lambda k, l: self.__call__(
                    *(_rnidxs(nidxs, paidxs[0], allidxs, k + 1, l + 1))
                ),
            )
        return M

    def _set_index_types(self, kwargs):
        if 'index_types' in kwargs:
            if (
                isinstance(kwargs['index_types'], (list, tuple))
                and len(kwargs['index_types']) == self.rank
                and all(
                    kwargs['index_types'][i] in [-1, 0, 1]
                    for i in range(self.rank)
                )
            ):
                self.index_types = kwargs['index_types']
            else:
                logger.error(f'Incorrect index_types list')
                raise GraviPyError('Incorrect index_types list')
        else:
            self.index_types = [0] * self.rank

    def _proper_tensor_indexes(self, *idxs):
        logger.debug(f'передаем индексы {idxs[0]=}')
        prank = len(idxs[0]) == self.rank
        logger.debug(f'{prank=}')
        logger.debug(
            [
                idxs[0][i] in self.index_values[self.index_types[i]]
                or (
                    isinstance(abs(idxs[0][i]), AllIdx)
                    and idxs[0][i] / abs(idxs[0][i]) + self.index_types[i] != 0
                )
                for i in range(self.rank)
            ]
        )
        if prank and all(
            [
                idxs[0][i] in self.index_values[self.index_types[i]]
                for i in range(self.rank)
            ]
        ):
            return 'component_mode'
        elif prank and all(
            [
                idxs[0][i] in self.index_values[self.index_types[i]]
                or (
                    isinstance(abs(idxs[0][i]), AllIdx)
                    and idxs[0][i] / abs(idxs[0][i]) + self.index_types[i] != 0
                )
                for i in range(self.rank)
            ]
        ):
            return 'matrix_mode'
        else:
            logger.error(
                f'Tensor component {self.symbol=} {idxs[0]=} does not exist'
            )
            raise GraviPyError(
                'Tensor component '
                + str(self.symbol)
                + str(idxs[0])
                + " doesn't  exist"
            )

    def _proper_derivative_indexes(self, *idxs):
        logger.debug(f'{idxs=}')
        if all(
            [idxs[0][i] in self.index_values[1] for i in range(len(idxs[0]))]
        ):
            return True
        else:
            logger.error(f'Derivative component {idxs[0]=} does not exist')
            raise GraviPyError(
                'Derivative component ' + str(idxs[0]) + " doesn't  exist"
            )

    def _connection_required(self, metric):
        logger.debug(f'{metric=}')
        if metric.conn is None or not isinstance(metric.conn, Christoffel):
            logger.error(
                f'Christoffel object for metric {metric.symbol=} is required'
            )
            raise GraviPyError(
                'Christoffel object for metric '
                + str(metric.symbol)
                + ' is required'
            )

    @staticmethod
    def get_nmatrixel(M, idxs):
        logger.debug(f'{M=}\n{idxs=}')
        if not isinstance(idxs, (tuple)):
            logger.error(f'Incorrect "idxs" parameter')
            raise GraviPyError('Incorrect "idxs" parameter')
        idxl = list(idxs)
        if len(idxl) == 1:
            return M[idxl[0]]
        elif len(idxl) == 2:
            return M[idxl[0], idxl[1]]
        else:
            if len(idxl) % 2 == 1:
                idx = idxl.pop(0)
                return Tensor.get_nmatrixel(M[idx], tuple(idxl))
            else:
                idx1 = idxl.pop(0)
                idx2 = idxl.pop(0)
                return Tensor.get_nmatrixel(M[idx1, idx2], tuple(idxl))

    def _compute_covariant_component(self, idxs):
        logger.debug(f'{idxs=}')
        if len(idxs) == 0:
            component = Function(str(self.symbol))(*self.coords.c)
        elif len(idxs) == 1:
            component = Function(str(self.symbol) + '(' + str(idxs[0]) + ')')(
                *self.coords.c
            )
        else:
            component = Function(str(self.symbol) + str(idxs))(*self.coords.c)
        return component

    def _compute_exttensor_component(self, idxs):
        logger.debug(f'{idxs=}')
        idxdict = dict(enumerate(idxs))
        idxargs = dict(enumerate(idxs))
        idxargs.update(
            dict(
                {
                    (i, 'c' + str(i))
                    for i in range(len(idxs))
                    if sympify(idxs[i]).is_negative
                }
            )
        )
        rl = [k for k in idxdict if sympify(idxdict[k]).is_negative]
        ii = tuple([list(idxdict.values())[i] for i in rl])
        ij = tuple([list(idxargs.values())[i] for i in rl])
        tsum = 0
        for ij in list(variations(range(1, self.dim + 1), len(ij), True)):
            idxargs.update(zip(rl, ij))
            tmul = self(*idxargs.values())
            for i in range(len(ii)):
                tmul = tmul * self.metric(ii[i], -(ij[i]))
            tsum = tsum + tmul
        component = tsum.together()
        self.components.update({idxs: component})
        return component

    def partialD(self, *idxs):
        logger.debug(f'{idxs=}')
        idxs = tuple(map(int, idxs))
        if len(idxs) <= self.rank:
            logger.error('Number of indexes must be greater than tensor rank')
            raise GraviPyError(
                'Number of indexes must be greater' + ' than tensor rank'
            )
        tidxs = idxs[0 : self.rank]
        didxs = idxs[self.rank :]
        self._proper_tensor_indexes(tidxs)
        self._proper_derivative_indexes(didxs)
        if idxs in self.partial_derivative_components.keys():
            return self.partial_derivative_components[idxs]
        else:
            component = self(*tidxs).diff(
                *map(self.coords, map(lambda x: -x, didxs))
            )
            self.partial_derivative_components.update({idxs: component})
            return component

    def covariantD(self, *idxs):
        logger.debug(f'{idxs=}')
        self._connection_required(self.metric)
        idxs = tuple(map(int, idxs))
        if len(idxs) <= self.rank:
            logger.error(f'Number of indexes must be greater than tensor rank')
            raise GraviPyError(
                'Number of indexes must be greater' + ' than tensor rank'
            )
        tidxs = idxs[0 : self.rank]
        didxs = idxs[self.rank :]
        self._proper_tensor_indexes(tidxs)
        self._proper_derivative_indexes(didxs)
        if idxs in self.covariant_derivative_components.keys():
            return self.covariant_derivative_components[idxs]
        else:
            nidxs = list(idxs)
            cidx = nidxs.pop(-1)
            if len(didxs) == 1:
                component = self(*tidxs).diff(self.coords(-cidx))
                for i in range(len(tidxs)):
                    sgn = tidxs[i] / abs(tidxs[i])
                    ci = dict(enumerate(tidxs))
                    for k in range(1, self.dim + 1):
                        ci.update({i: sgn * k})
                        cil = tuple(ci.values())
                        if tidxs[i] > 0:
                            component = component - self.metric.conn(
                                -k, tidxs[i], cidx
                            ) * self(*cil)
                        else:
                            component = component + self.metric.conn(
                                tidxs[i], k, cidx
                            ) * self(*cil)
            else:
                component = self.covariantD(*nidxs).diff(self.coords(-cidx))
                for i in range(len(nidxs)):
                    sgn = nidxs[i] / abs(nidxs[i])
                    ci = dict(enumerate(nidxs))
                    for k in range(1, self.dim + 1):
                        ci.update({i: sgn * k})
                        cil = tuple(ci.values())
                        if nidxs[i] > 0:
                            component = component - self.metric.conn(
                                -k, nidxs[i], cidx
                            ) * self.covariantD(*cil)
                        else:
                            component = component + self.metric.conn(
                                nidxs[i], k, cidx
                            ) * self.covariantD(*cil)
            component = component.together()
            self.covariant_derivative_components.update({idxs: component})
            return component


class Coordinates(GeneralTensor):
    r"""
    Represents a class of Coordinate n-vectors.

    Parameters
    ==========

    symbol : python string - name of the Coordinate n-vector
    coords : list of SymPy Symbol objects - list of coordinates

    Examples
    ========

    Define a Coordinate 4-vector:

    >>> from tensorial2 import *
    >>> t, r, theta, phi = sympy.symbols('t, r, \\theta, \phi')
    >>> chi = Coordinates('\chi', [t, r, theta, phi])
    >>> chi(-1)
    t
    >>> chi(-3)
    \theta

    Python list representation of the coordinate tensor chi

    >>> chi.c
    [t, r, \theta, \phi]

    and it's SymPy Matrix representation

    >>> chi(-All)
    Matrix([[t, r, \theta, \phi]])

    >>> chi.components
    {(-1,): t, (-2,): r, (-3,): \theta, (-4,): \phi}

    Covariant components are not defined until MetricTensor object is not
    created

    >>> chi(1) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    GraviPyError: "Tensor component \\chi(1,) doesn't  exist"

    """

    def __init__(self, symbol, coords):
        super().__init__(symbol, 1, coords, index_types=[-1])
        self.is_coordinate_tensor = True
        self.coords = self
        self.c = coords
        self.components = {(-i - 1,): coords[i] for i in range(self.dim)}

    def __len__(self):
        return self.dim


class MetricTensor(GeneralTensor):
    r"""
    Represents a class of Metric Tensors.

    Parameters
    ==========

    symbol : python string - name of the Coordinate n-vector
    coords : GraviPy Coordinates object
    metric : SymPy Matrix object - metric tensor components in ``coords`` sytem

    Examples
    ========

    Define the Schwarzshild MetricTensor:

    >>> from gravipy.tensorial import *
    >>> t, r, theta, phi = symbols('t, r, \\theta, \phi')
    >>> chi = Coordinates('\chi', [t, r, theta, phi])
    >>> M = Symbol('M')
    >>> Metric = diag(-(1 - 2 * M / r), 1 / (1 - 2 * M / r), r ** 2,
    ...                  r ** 2 * sin(theta) ** 2)
    >>> g = MetricTensor('g', chi, Metric)

    SymPy matrix representation of the metric tensor

    >>> Matrix(4, 4, lambda i, j: g(i + 1, j + 1))
    Matrix([
    [2*M/r - 1,              0,    0,                   0],
    [        0, 1/(-2*M/r + 1),    0,                   0],
    [        0,              0, r**2,                   0],
    [        0,              0,    0, r**2*sin(\theta)**2]])

    or for short

    >>> g(All, All)
    Matrix([
    [2*M/r - 1,              0,    0,                   0],
    [        0, 1/(-2*M/r + 1),    0,                   0],
    [        0,              0, r**2,                   0],
    [        0,              0,    0, r**2*sin(\theta)**2]])

    Contravariant and mixed tensor component

    >>> g(-All, -All)
    Matrix([
    [1/(2*M/r - 1),          0,       0,                       0],
    [            0, -2*M/r + 1,       0,                       0],
    [            0,          0, r**(-2),                       0],
    [            0,          0,       0, 1/(r**2*sin(\theta)**2)]])

    >>> g(All, -All)
    Matrix([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]])

    Contravariant and covariant components of Coordinate 4-vector

    >>> chi(-All)
    Matrix([[t, r, \theta, \phi]])

    >>> chi(All)
    Matrix([[t*(2*M/r - 1), r/(-2*M/r + 1), \theta*r**2, \phi*r**2*sin(\theta)**2]])

    """

    def __init__(self, symbol, coords, metric):
        super().__init__(symbol, 2, coords)
        self.is_metric_tensor = True
        self.coords = coords
        self.metric = self
        logger.debug(f'{self.metric=}')
        logger.debug(f'{metric**-1=}')
        logger.debug(f'{self.index_values=}')
        self.components.update(
            {
                (i, j): metric[i - 1, j - 1]
                for i in self.index_values[1]
                for j in self.index_values[1]
            }
        )
        logger.debug(f'#1 {self.components=}')
        self.components.update(
            {
                (i, j): (metric**-1)[-i - 1, -j - 1]
                for i in self.index_values[-1]
                for j in self.index_values[-1]
            }
        )
        logger.debug(f'#2 {self.components=}')
        self.components.update(
            {
                (i, j): sympy.KroneckerDelta(abs(i), abs(j))
                for i in self.index_values[0]
                for j in self.index_values[0]
                if i * j < 0
            }
        )
        logger.debug(f'#3 {self.components=}')
        logger.debug(f'{coords.metric=}')
        coords.metric = self
        logger.debug(f'{coords.metric=}')
        coords.index_types = [0]
        coords.components.update(
            {
                (i,): sum(
                    self.components[i, j] * self.coords(-j)
                    for j in self.index_values[1]
                )
                for i in self.index_values[1]
            }
        )
        logger.debug(f'#4 {self.components=}')


class GraviPyError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class AllIdx(sympy.Symbol):
    def __new__(cls, name, **kwargs):
        logger.debug(f'вызов для {cls} с параметрами {name} и {kwargs}\n')
        return super().__new__(cls, name, **kwargs)


All = AllIdx('All', positive=True)


if __name__ == '__main__':
    # import doctest

    # doctest.testmod()

    from sympy import init_printing

    init_printing()

    t, r, theta, phi = sympy.symbols('t, r, \\theta, \phi')

    chi = Coordinates(
        '\\chi',  # symbol
        [t, r, theta, phi],  # coords
    )

    M = sympy.Symbol('M')
    Metric = sympy.diag(
        -(1 - 2 * M / r),
        1 / (1 - 2 * M / r),
        r**2,
        r**2 * sympy.sin(theta) ** 2,
    )

    g = MetricTensor('g', chi, Metric)

    # print(chi(-1))
    # print('CHI minusALL\n')
    # print(chi(-All))

    # print(chi(All))

    # print(chi.components)
