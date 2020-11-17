"""
    Set of utilities for writing code generation info to json file (that will be then loaded at runtime).

    These are standalone methods
"""

from codegen import json_definitions as jd
from codegen import fblas_routine
from codegen import fblas_types
from codegen.fblas_types import RoutineType
import json


def write_to_file(path: str, json_content):
    with open(path, 'w') as json_file:
        json.dump(json_content, json_file, indent=4)



def add_commons(json:dict, routine:fblas_routine.FBLASRoutine):
    """
    Adds fields that are common to all routines: routine type, user_name, data type,...
    :param json: 
    :param routine:
    """
    json[jd.BLAS_NAME_KEY] = routine.blas_name
    json[jd.USER_NAME_KEY] = routine.user_name

    #Precision
    #TODO: support generic type also at runtime
    if routine.type is RoutineType.Float:
        json[jd.PRECISION_KEY] = "single"
    elif routine.type is RoutineType.Double:
        json[jd.PRECISION_KEY] = "double"
    else:
        raise TypeError("Routine type not fully supported")

    if routine.has_2D_computational_tile:
        json[jd.WIDTH_X_KEY] = int(routine.width_x)
        json[jd.WIDTH_Y_KEY] = int(routine.width_y)
    else:
        json[jd.WIDTH_KEY] = int(routine.width)




def add_incx(json: dict, routine: fblas_routine.FBLASRoutine):
    json[jd.INCX_KEY] = routine.incx


def add_incy(json: dict, routine: fblas_routine.FBLASRoutine):
    json[jd.INCY_KEY] = routine.incy


def add_tile_n_size(json: dict, routine: fblas_routine.FBLASRoutine):
    json[jd.TILE_N_SIZE_KEY] = routine.tile_n_size


def add_tile_m_size(json: dict, routine: fblas_routine.FBLASRoutine):
    json[jd.TILE_M_SIZE_KEY] = routine.tile_m_size

def add_tile_size(json: dict, routine: fblas_routine.FBLASRoutine):
    json[jd.TILE_SIZE_KEY] = routine.tile_size


def add_transposed(json: dict, routine: fblas_routine.FBLASRoutine):
    if routine.transposedA is fblas_types.FblasTranspose.FblasNoTrans:
        json[jd.TRANS_A_KEY]="N"
    else:
        json[jd.TRANS_A_KEY] = "T"

def add_transposedB(json: dict, routine: fblas_routine.FBLASRoutine):
    if routine.transposedB is fblas_types.FblasTranspose.FblasNoTrans:
        json[jd.TRANS_B_KEY]="N"
    else:
        json[jd.TRANS_B_KEY] = "T"



def add_item(json:dict, key: str, value: str):
    """
    Adds a generic item to the json
    :param json:
    :param key:
    :param value:
    :return:
    """
    json[key] = value


def add_order(json: dict, routine: fblas_routine.FBLASRoutine):

    if routine.order == fblas_types.FblasOrder.FblasRowMajor:
        json[jd.ORDER_KEY] = "RowMajor"
    else:
        json[jd.ORDER_KEY] = "ColumnMajor"

def add_uplo(json: dict, routine: fblas_routine.FBLASRoutine):
    if routine.uplo == fblas_types.FblasUpLo.FblasUpper:
        json[jd.UPLO_KEY] = "U"
    else:
        json[jd.UPLO_KEY] = "L"

def add_side(json: dict, routine: fblas_routine.FBLASRoutine):
    if routine.side == fblas_types.FblasSide.FblasLeft:
        json[jd.SIDE_KEY] = "L"
    else:
        json[jd.SIDE_KEY] = "R"