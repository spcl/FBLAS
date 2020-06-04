"""
   Fblas helper representation
"""

from codegen import fblas_types
from codegen import generator_definitions
class FBLASHelper:

    #name of the helper
    _helper_name = ""
    _user_name = ""

    # Type of the helper
    _type: fblas_types.RoutineType
    _type_str: str

    # Channel name: name of input/output channel (depends on the helper)
    _channel_name: str

    # spatial parallelism (vectorization width, used to unroll channels reads/writes)
    _width =  generator_definitions.DEFAULT_WIDTH

    # strides: for helper they will be the same so that we can use any of these two fields
    _incx = 1
    _incy = 1

    # Level 2/3: tile sizes
    _tile_n_size = 0
    _tile_m_size = 0

    # Tiles and element order (for helpers that deal with matrices)
    _tiles_order: fblas_types.FblasOrder
    _elements_order: fblas_types.FblasOrder

    def __init__(self, helper_name: str,  user_name: str, chan_name: str, type: fblas_types.RoutineType):
        self._helper_name = helper_name
        self._user_name = user_name
        self._channel_name = chan_name
        self._type = type
        self._type_str = fblas_types.ROUTINE_TYPE_TO_TYPE_STR[type]

    @property
    def helper_name(self):
        return self._helper_name

    @property
    def user_name(self):
        return self._user_name

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, width: int):
        self._width = width

    @property
    def type_str(self):
        return self._type_str

    @property
    def stride(self):
        return self._incx

    @stride.setter
    def stride(self, stride: int):
        self._incx = stride
        self._incy = stride

    @property
    def incx(self):
        return self._incx

    @incx.setter
    def incx(self, incx: int):
        self._incx = incx

    @property
    def incy(self):
        return self._incy

    @incy.setter
    def incx(self, incy: int):
        self._incy = incy

    @property
    def channel_name(self):
        return self._channel_name

    @channel_name.setter
    def channel_name(self, chan_name : str):
        self._channel_name = chan_name

    @property
    def tile_n_size(self):
        return self._tile_n_size

    @tile_n_size.setter
    def tile_n_size(self, tile_size: int):
        self._tile_n_size = tile_size

    @property
    def tile_m_size(self):
        return self._tile_m_size

    @tile_m_size.setter
    def tile_m_size(self, tile_size: int):
        self._tile_m_size = tile_size

    @property
    def elements_order(self):
        return self._elements_order

    @elements_order.setter
    def elements_order(self, order):
        self._elements_order = order

    @property
    def tiles_order(self):
        return self._tiles_order

    @tiles_order.setter
    def tiles_order(self, order):
        self._tiles_order = order
