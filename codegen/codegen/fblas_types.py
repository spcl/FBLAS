"""
    Collection of type definitions
"""



import aenum

class RoutineType(aenum.AutoNumberEnum):
    """ Available type for FBLAS routines """

    Float = (),                 # Single Precision
    Double = ()                 # Double Precision

class Platform(aenum.AutoNumberEnum):
    """ Supported Platform """
    Stratix10 = (),
    Arria10 = ()


TYPE_STR_TO_ROUTINE_TYPE ={
    "float" : RoutineType.Float,
    "double": RoutineType.Double
}

ROUTINE_TYPE_TO_TYPE_STR ={
    RoutineType.Float : "float",
    RoutineType.Double: "double"
}

# Size of shift register according to platform  and type
SHIFT_REGISTER_SIZES = {
    (RoutineType.Float, Platform.Stratix10): 64,
    (RoutineType.Double, Platform.Stratix10): 64
}

class FblasCodegen(aenum.AutoNumberEnum):
    """ Indicates the codegen used """
    HostCodegen = (),
    ModulesCodegen = ()

class FblasOrder(aenum.AutoNumberEnum):
    """ Matrix order """
    FblasRowMajor = (),
    FblasColMajor = (),
    FblasOrderUndef = ()

class FblasTranspose(aenum.AutoNumberEnum):
    """ Matrix transposition value"""
    FblasNoTrans = (),
    FblasTrans = (),
    FblasTransUndef = ()


class FblasUpLo(aenum.AutoNumberEnum):
    """ Upper Lower Matrix"""
    FblasUpper = (),
    FblasLower = (),
    FblasUpLoUndef = ()

class FblasSide(aenum.AutoNumberEnum):
    FblasLeft = (),
    FblasRight = (),
    FblasSideUndef = ()

class FblasDiag(aenum.AutoNumberEnum):
    FblasUnit = (),
    FblasNoUnit = (),
    FblasDiagUndef = ()
