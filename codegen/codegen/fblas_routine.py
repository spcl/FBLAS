"""
    FBlas Routine class: it used to represent a routine definition, specified by the user using JSON file.
    It is used by the Host and Module Codegen (specified by the _codegen variable). Accordingly,
    some class members could be invalid.

"""

from codegen import fblas_types
from codegen import generator_definitions

class FBLASRoutine:
    # name of the routine according to blas (without indication of the precision)
    _blas_name = ""

    # user name for the routine
    _user_name = ""

    # spatial parallelism (vectorization width)
    _width = generator_definitions.DEFAULT_WIDTH

    # data type used in routine
    _type: fblas_types.RoutineType
    _type_str: str

    # if the routine has to use shift registers (e.g. double precision) or not
    # and in case how big they should be
    _uses_shift_registers = False
    _size_shift_registers = 0


    # The type of codegen:Host/Modules
    _codegen = None

    # inx/incy
    _incx = 1
    _incy = 1

    # Level 2/3: tile sizes
    _tile_n_size = generator_definitions.DEFAULT_TILE_SIZE
    _tile_m_size = generator_definitions.DEFAULT_TILE_SIZE

    # Matrix characteristics
    _order = None
    _diag = None
    _transposeA = None
    _transposeB = None
    _side = None
    _uplo = None

    # input/output channels (useful for Module Codegen)
    # these are instance member dictionaries "required_channel_name" -> "user_name"
    _input_channels = None
    _output_channels = None

    # Tiles and element order (for level2/3 that works with matrices)
    # The order is RowMajor if tiles/element are row streamed
    # otherwise it is ColumnMajor
    _tiles_A_order: fblas_types.FblasOrder = fblas_types.FblasOrder.FblasRowMajor
    _elements_A_order: fblas_types.FblasOrder = fblas_types.FblasOrder.FblasRowMajor

    # Indicates whether or not this routines has a 2D computatioal tile (e.g. GEMM)
    _has_2D_computational_tile = False
    # If yes, there are the two vectorization width
    _width_x = 0
    _width_y = 0
    _tile_size = 0
    _systolic = False
    _vect_size = 0



    def __init__(self, blas_name: str, user_name: str, type: fblas_types.RoutineType, platform: fblas_types.Platform, codegen: fblas_types.FblasCodegen):
        self._blas_name = blas_name
        self._user_name = user_name
        self._type = type
        self._type_str = fblas_types.ROUTINE_TYPE_TO_TYPE_STR[type]
        self._platform = platform
        self._codegen = codegen
        self._width = generator_definitions.DEFAULT_WIDTH
        # Declare all the instance variables
        self._input_channels = {}
        self._output_channels = {}
        self._incx = 1
        self._incy = 1
        self._tile_n_size = generator_definitions.DEFAULT_TILE_SIZE
        self._tile_m_size = generator_definitions.DEFAULT_TILE_SIZE
        self._order = fblas_types.FblasOrder.FblasOrderUndef
        self._diag = fblas_types.FblasDiag.FblasDiagUndef
        self._transposeA = fblas_types.FblasTranspose.FblasTransUndef
        self._transposeB = fblas_types.FblasTranspose.FblasTransUndef
        self._side = fblas_types.FblasSide.FblasSideUndef
        self._uplo = fblas_types.FblasUpLo.FblasUpLoUndef
        if type == fblas_types.RoutineType.Double:
            self._uses_shift_registers = True
            self._size_shift_registers = fblas_types.SHIFT_REGISTER_SIZES[(type, platform)]
        else:
            self._uses_shift_registers = False
        self._has_2D_computational_tile = False
        self._width_x = self._width = generator_definitions.DEFAULT_2D_CTILE_WIDTH
        self._width_y = self._width = generator_definitions.DEFAULT_2D_CTILE_WIDTH
        self._tile_size = generator_definitions.DEFAULT_TILE_SIZE
        self._systolic = False
        self._vect_size = 4


    def __str__(self):
        return """Routine {} implements {} with type {}
        Width: {} Incx: {} Incy: {}""".format(self._user_name, self._blas_name, self._type, self._width, self._incx, self._incy)


    #Getter/setter
    @property
    def blas_name(self):
        return self._blas_name

    @property
    def user_name(self):
        return self._user_name

    @property
    def type(self):
        return self._type

    @property
    def type_str(self):
        return self._type_str

    @property
    def uses_shift_registers(self):
        return self._uses_shift_registers

    @uses_shift_registers.setter
    def uses_shift_registers(self, value: bool):
        #if the routine uses shift register, set the size
        self._uses_shift_registers = value
        if value:
            self._size_shift_registers = fblas_types.SHIFT_REGISTER_SIZES[(self.type, self._platform)]

    @property
    def size_shift_registers(self):
        return self._size_shift_registers


    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, width: int):
        self._width = width

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
    def incy(self, incy: int):
        self._incy = incy

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
    def tile_size(self):
        return self._tile_size

    @tile_size.setter
    def tile_size(self, tile_size: int):
        self._tile_size = tile_size

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, order: fblas_types.FblasOrder):
        self._order = order

    @property
    def uplo(self):
        return self._uplo

    @uplo.setter
    def uplo(self, uplo: fblas_types.FblasUpLo):
        self._uplo = uplo

    @property
    def transposedA(self):
        return self._transposeA

    @transposedA.setter
    def transposedA(self, trans: fblas_types.FblasTranspose):
        self._transposeA = trans

    @property
    def transposedB(self):
        return self._transposeB

    @transposedB.setter
    def transposedB(self, trans: fblas_types.FblasTranspose):
        self._transposeB = trans

    @property
    def input_channels(self):
        return self._input_channels

    @property
    def output_channels(self):
        return self._output_channels

    @property
    def tiles_A_order(self):
        return self._tiles_A_order

    @tiles_A_order.setter
    def tiles_A_order(self, order: fblas_types.FblasOrder):
        self._tiles_A_order = order

    @property
    def elements_A_order(self):
        return self._elements_A_order

    @elements_A_order.setter
    def elements_A_order(self, order : fblas_types.FblasOrder):
        self._elements_A_order = order

    @property
    def has_2D_computational_tile(self):
        return self._has_2D_computational_tile

    @has_2D_computational_tile.setter
    def has_2D_computational_tile(self, value: bool):
        self._has_2D_computational_tile = value

    @property
    def width_x(self):
        return self._width_x

    @width_x.setter
    def width_x(self, width: int):
        self._width_x = width

    @property
    def width_y(self):
        return self._width_y

    @width_y.setter
    def width_y(self, width: int):
        self._width_y = width

    @property
    def systolic(self):
        return self._systolic

    @systolic.setter
    def systolic(self, value: bool):
        self._systolic = value

    @property
    def vect_size(self):
        return self._vect_size

    @vect_size.setter
    def vect_size(self, value: int):
        self._vect_size = value



    def are_tiles_A_rowstreamed(self):
        """
        :return: True if the tiles of A are rowstreamed
        """
        return self._tiles_A_order == fblas_types.FblasOrder.FblasRowMajor

    def are_elements_A_rowstreamed(self):
        """
        :return: True if the elements of A are rowstreamed
        """
        return self._elements_A_order == fblas_types.FblasOrder.FblasRowMajor


    def add_input_channel(self, routine_channel_name, user_name):
        '''
        Add the channel to the dictionary of input channels
        If already present, it will be overwritten
        '''
        self._input_channels[routine_channel_name] = user_name

    def add_output_channel(self, routine_channel_name, user_name):
        '''
        Add the channel to the dictionary of input channels
        If already present, it will be overwritten
        '''
        self._output_channels[routine_channel_name] = user_name

