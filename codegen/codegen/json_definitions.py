"""
    This file contains a list of constant definitions used for parsing/unparsing from json
"""

ROUTINE_KEY = "routine"
PLATFORM_KEY = "platform"
TYPE_KEY = "type"
BLAS_NAME_KEY = "blas_name"
USER_NAME_KEY = "user_name"
PRECISION_KEY = "precision" #used for output json
HELPER_KEY = "helper"
HELPER_NAME_KEY = "helper_name"
HELPER_CHANNEL_NAME_KEY = "channel_name"


########################
# HOST API GENERATOR KEYS
########################

WIDTH_KEY = "width"
INCX_KEY = "incx"
INCY_KEY = "incy"
TILE_SIZE_KEY = "tile size"
TILE_N_SIZE_KEY = "tile N size"
TILE_M_SIZE_KEY = "tile M size"
TRANS_KEY = "trans"
TRANS_A_KEY = "transa"
TRANS_B_KEY = "transb"
UPLO_KEY = "uplo"
SIDE_KEY = "side"
ORDER_KEY = "order"
TILES_A_ORDER_KEY = "A tiles order"
ELEMENTS_A_ORDER_KEY = "A elements order"
WIDTH_X_KEY = "width x"
WIDTH_Y_KEY = "width y"
SYSTOLIC_KEY = "systolic"
VECT_SIZE_KEY = "vect size"


# used for helpers
STRIDE_KEY = "stride"
TILES_ORDER_KEY = "tiles order"
ELEMENTS_ORDER_KEY = "elements order"


#######################
# MODULE GENERATOR KEYS
#######################

LDA_KEY = "lda"
LDB_KEY = "ldb"

#########################
# GENERATED JSON FIELDS
########################
GENERATED_READ_VECTOR_X = "read_vector_x"
GENERATED_READ_VECTOR_Y = "read_vector_y"
GENERATED_READ_VECTOR_X_TRANS = "read_vector_x_trans"
GENERATED_READ_VECTOR_Y_TRANS = "read_vector_y_trans"
GENERATED_READ_VECTOR_X_TRSV = "read_vector_x_trsv"
GENERATED_WRITE_SCALAR = "write_scalar"
GENERATED_WRITE_VECTOR = "write_vector"
GENERATED_READ_MATRIX_A = "read_matrix_A"
GENERATED_READ_MATRIX_A2 = "read_matrix_A2"
GENERATED_READ_MATRIX_B = "read_matrix_B"
GENERATED_READ_MATRIX_B2 = "read_matrix_B2"
GENERATED_WRITE_MATRIX = "write_matrix"
GENERATED_TILE_N_SIZE = "tile N size"
GENERATED_TILE_M_SIZE = "tile M size"
GENERATED_MTILE_SIZE = "tile size"
GENERATED_ORDER = "order"
GENERATED_TRANSPOSED_A = "transA"
GENERATED_TRANSPOSED_B = "transB"
GENERATED_UPLO = "uplo"
GENERATED_SIDE = "side"
GENERATED_SYSTOLIC = "systolic"

##########################
# Default values
##########################
# TODO: don't know if needed


###############################
# Routine definition conf file
###############################

RD_NAME_KEY = "name"
RD_REQUIRED_KEY = "required_parameters"
RD_OPTIONAL_KEY = "optional_parameters"
RD_REQUIRED_INPUTS = "required_inputs"
RD_REQUIRED_OUTPUTS = "required_outputs"
RD_OPTIONAL_OUTPUTS = "optional_outputs"


DEFAULT_PLATFORM = "Stratix 10"
STRATIX10 = "Stratix 10"
ARRIA10 = "Arria 10"

##############################
# Paths
##############################

PATH_HOST_API_RD = "conf/routine_defs_host_api.json"
PATH_MODULES_RD = "conf/routine_defs_modules_codegen.json"
PATH_HELPERS_D = "conf/helper_defs_modules_codegen.json"

