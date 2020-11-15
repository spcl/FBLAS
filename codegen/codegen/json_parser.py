"""

    Set of utilities for parsing/unparsing json files that can be input for fblas host/module generator
"""

import json
from codegen import json_definitions as jd
from codegen import fblas_routine
from codegen import fblas_types
from codegen import fblas_helper
import logging
import os


class JSONParser:
    # The type of codegen that is using the parser
    _codegen = None
    _is_host_codegen = False

    # dictionaries of required and optional parameters for each BLAS routine (Host API)
    _host_api_required_parameters = {}
    _host_api_optional_parameters = {}
    _host_api_supported_routines = set()

    # dictionaries of required and optional parameters for each BLAS routine (Modules Codegen)
    _modules_codegen_required_parameters = {}
    _modules_codegen_optional_parameters = {}
    _modules_codegen_required_input_channels = {}
    _modules_codegen_required_output_channels = {}
    _modules_codegen_optional_output_channels = {
    }  # can be used for particular version of some routine
    _modules_codegen_supported_routines = set()
    # for the modules codegen we have to take care of helpers
    _modules_codegen_supported_helpers = set()
    _modules_codegen_helpers_required_parameters = {}
    _modules_codegen_helpers_optional_parameters = {}

    # routines that support 2D computational tiles
    _routines_with_2D_computational_tile = ["gemm", "syrk", "syr2k"]

    def __init__(self, codegen: fblas_types.FblasCodegen):
        if codegen == fblas_types.FblasCodegen.HostCodegen:
            self._load_routine_definitions_host_api()
        else:
            self._load_routine_definitions_modules_codegen()
            self._load_helper_definitions_modules_codegen()
        self._platform = fblas_types.Platform.Stratix10  # Default Platform
        self._codegen = codegen
        self._is_host_codegen = (
            codegen == fblas_types.FblasCodegen.HostCodegen)

    def _load_routine_definitions_host_api(self):
        '''
        Loads the routine definitions for the host API. This is a json file that, for each supported routines, contains
        the required and optional field
        '''
        file_path = os.path.join(os.path.dirname(__file__),
                                 jd.PATH_HOST_API_RD)
        with open(file_path) as json_file:
            document = json.load(json_file)

        for r in document[jd.ROUTINE_KEY]:
            name = r[jd.RD_NAME_KEY]
            self._host_api_supported_routines.add(name)

            # load required parameters if any
            if self._is_json_key_present(r, jd.RD_REQUIRED_KEY):
                required_pars = []
                for p in r[jd.RD_REQUIRED_KEY]:
                    required_pars.append(p)
                self._host_api_required_parameters[name] = required_pars

            # load optional parameters if any
            if self._is_json_key_present(r, jd.RD_OPTIONAL_KEY):
                optional_pars = []
                for p in r[jd.RD_OPTIONAL_KEY]:
                    optional_pars.append(p)
                self._host_api_optional_parameters[name] = optional_pars

    def _load_routine_definitions_modules_codegen(self):
        '''
        Loads the routine definition for modules codegen. The json file contains, for each supported routines,
        the required and optional field
        '''
        file_path = os.path.join(os.path.dirname(__file__), jd.PATH_MODULES_RD)
        with open(file_path) as json_file:
            document = json.load(json_file)

        for r in document[jd.ROUTINE_KEY]:
            name = r[jd.RD_NAME_KEY]
            self._modules_codegen_supported_routines.add(name)

            # load required parameters if any
            if self._is_json_key_present(r, jd.RD_REQUIRED_KEY):
                required_pars = []
                for p in r[jd.RD_REQUIRED_KEY]:
                    required_pars.append(p)
                self._modules_codegen_required_parameters[name] = required_pars

            # load optional parameters if any
            if self._is_json_key_present(r, jd.RD_OPTIONAL_KEY):
                optional_pars = []
                for p in r[jd.RD_OPTIONAL_KEY]:
                    optional_pars.append(p)
                self._modules_codegen_optional_parameters[name] = optional_pars

            # load required input and output channels
            if self._is_json_key_present(r, jd.RD_REQUIRED_INPUTS):
                required_in_chans = []
                for c in r[jd.RD_REQUIRED_INPUTS]:
                    required_in_chans.append(c)
                self._modules_codegen_required_input_channels[
                    name] = required_in_chans

            if self._is_json_key_present(r, jd.RD_REQUIRED_OUTPUTS):
                required_out_chans = []
                for c in r[jd.RD_REQUIRED_OUTPUTS]:
                    required_out_chans.append(c)
                self._modules_codegen_required_output_channels[
                    name] = required_out_chans

            # load optional input channels:
            # TODO if needed

            # load optional output channels: it could be required for some routines
            if self._is_json_key_present(r, jd.RD_OPTIONAL_OUTPUTS):
                optional_out_chans = []
                for c in r[jd.RD_OPTIONAL_OUTPUTS]:
                    optional_out_chans.append(c)
                self._modules_codegen_optional_output_channels[
                    name] = optional_out_chans

    def _load_helper_definitions_modules_codegen(self):
        '''
        Loads the helper definition for modules codegen. The json file contains, for each supported helper,
        the required and optional field
        '''
        file_path = os.path.join(os.path.dirname(__file__), jd.PATH_HELPERS_D)
        with open(file_path) as json_file:
            document = json.load(json_file)

        for r in document[jd.HELPER_KEY]:
            name = r[jd.RD_NAME_KEY]
            self._modules_codegen_supported_helpers.add(name)

            # load required parameters if any
            if self._is_json_key_present(r, jd.RD_REQUIRED_KEY):
                required_pars = []
                for p in r[jd.RD_REQUIRED_KEY]:
                    required_pars.append(p)
                self._modules_codegen_helpers_required_parameters[
                    name] = required_pars

            # load optional parameters if any
            if self._is_json_key_present(r, jd.RD_OPTIONAL_KEY):
                optional_pars = []
                for p in r[jd.RD_OPTIONAL_KEY]:
                    optional_pars.append(p)
                self._modules_codegen_helpers_optional_parameters[
                    name] = optional_pars

    def _is_json_key_present(self, json: dict, key: str):
        '''
        Checks whether a key is present in a json dictionary
        :param json: json
        :param key: name of the key
        :return: true if the key is present, false otherwise
        '''
        try:
            buf = json[key]
        except KeyError:
            return False

        return True

    def _parse_width(self, json: dict, routine):
        """ Parses the width attribute """
        width = json[jd.WIDTH_KEY]
        if not isinstance(width, int) or width < 0:
            logging.warning(
                "Routine {}, defined width must be a number greater than zero".
                format(routine.user_name))
            return False
        else:
            routine.width = width
            return True

    def _parse_width_x(self, json: dict, routine):
        """ Parses the width_x attribute (routines with 2D computational tiles)"""
        width = json[jd.WIDTH_X_KEY]
        if not isinstance(width, int) or width < 0:
            logging.warning(
                "Routine {}, defined width x must be a number greater than zero"
                .format(routine.user_name))
            return False
        else:
            routine.width_x = width
            return True

    def _parse_width_y(self, json: dict, routine):
        """ Parses the width_y attribute (routines with 2D computational tiles)"""
        width = json[jd.WIDTH_Y_KEY]
        if not isinstance(width, int) or width < 0:
            logging.warning(
                "Routine {}, defined width y must be a number greater than zero"
                .format(routine.user_name))
            return False
        else:
            routine.width_y = width
            return True

    def _parse_incx(self, json: dict, routine: fblas_routine.FBLASRoutine):
        incx = json[jd.INCX_KEY]
        if not isinstance(incx, int):
            logging.warning(
                "Routine {}, defined incx is not a valid number".format(
                    routine.user_name))
            return False
        else:
            routine.incx = incx
            return True

    def _parse_incy(self, json: dict, routine: fblas_routine.FBLASRoutine):
        incy = json[jd.INCY_KEY]
        if not isinstance(incy, int):
            logging.warning(
                "Routine {}, defined incx is not a valid number".format(
                    routine.user_name))
            return False
        else:
            routine.incy = incy
            return True

    def _parse_tile_n_size(self, json: dict, routine):
        size = json[jd.TILE_N_SIZE_KEY]
        if not isinstance(size, int) and size < 0:
            logging.warning(
                "Routine {}, defined tile N size is not a valid number".format(
                    routine.user_name))
            return False
        else:
            routine.tile_n_size = size
            return True

    def _parse_tile_m_size(self, json: dict, routine):
        size = json[jd.TILE_M_SIZE_KEY]
        if not isinstance(size, int) and size < 0:
            logging.warning(
                "Routine {}, defined tile M size is not a valid number".format(
                    routine.user_name))
            return False
        else:
            routine.tile_m_size = size
            return True

    def _parse_tile_size(self, json: dict, routine):
        size = json[jd.TILE_SIZE_KEY]
        if not isinstance(size, int) and size < 0:
            logging.warning(
                "Routine {}, defined tile size is not a valid number".format(
                    routine.user_name))
            return False
        else:
            routine.tile_size = size
            return True

    def _parse_systolic(self, json: dict, routine):
        # systolic is a boolean flag
        systolic = json[jd.SYSTOLIC_KEY]
        if not isinstance(systolic, bool):
            logging.warning(
                "Routine {}, defined systolic must be a boolean value".format(
                    routine.user_name))
            return False
        else:
            routine.systolic = systolic
            return True

    def _parse_trans(self, json: dict, routine):
        trans = json[jd.TRANS_KEY]
        if trans.lower() == 'n':
            routine.transposedA = fblas_types.FblasTranspose.FblasNoTrans
            return True
        elif trans.lower() == 't':
            routine.transposedA = fblas_types.FblasTranspose.FblasTrans
            return True
        else:
            logging.warning(
                "Routine {}, defined trans is not valid. It should be 'n' (NoTrans) or 't' (Trans)"
                .format(routine.user_name))
            return False

    def _parse_transa(self, json: dict, routine):
        trans = json[jd.TRANS_A_KEY]
        if trans.lower() == 'n':
            routine.transposedA = fblas_types.FblasTranspose.FblasNoTrans
            return True
        elif trans.lower() == 't':
            routine.transposedA = fblas_types.FblasTranspose.FblasTrans
            return True
        else:
            logging.warning(
                "Routine {}, defined transa is not valid. It should be 'n' (NoTrans) or 't' (Trans)"
                .format(routine.user_name))
            return False

    def _parse_transb(self, json: dict, routine):
        trans = json[jd.TRANS_B_KEY]
        if trans.lower() == 'n':
            routine.transposedB = fblas_types.FblasTranspose.FblasNoTrans
            return True
        elif trans.lower() == 't':
            routine.transposedB = fblas_types.FblasTranspose.FblasTrans
            return True
        else:
            logging.warning(
                "Routine {}, defined transb is not valid. It should be 'n' (NoTrans) or 't' (Trans)"
                .format(routine.user_name))
            return False

    def _parse_order(self, json: dict, routine):
        order = json[jd.ORDER_KEY]
        if order.lower() == 'rowmajor':
            routine.order = fblas_types.FblasOrder.FblasRowMajor
            return True
        elif order.lower() == 'columnmajor':
            logging.warning(
                "Routine {}, currently only RowMajor order is supported in host API"
                .format(routine.user_name))
            return False
        else:
            logging.warning(
                "Routine {}, defined order is not valid. It should be 'rowmajor' ('columnmajor' not supported in this release)"
                .format(routine.user_name))
            return False

    def _parse_uplo(self, json: dict, routine):
        uplo = json[jd.UPLO_KEY]
        if uplo.lower() == "l":
            routine.uplo = fblas_types.FblasUpLo.FblasLower
            return True
        elif uplo.lower() == "u":
            routine.uplo = fblas_types.FblasUpLo.FblasUpper
            return True
        else:
            logging.warning(
                "Routine {}, defined uplo is not valid. It should be 'l' or 'u' "
                .format(routine.user_name))
            return False

    def _parse_elements_order(self, json: dict, helper):
        # Used for helpers only
        order = json[jd.ELEMENTS_ORDER_KEY]
        if order.lower() == 'row':
            helper.elements_order = fblas_types.FblasOrder.FblasRowMajor
            return True
        elif order.lower() == 'column':
            helper.elements_order = fblas_types.FblasOrder.FblasColMajor
            return True
        else:
            logging.warning(
                "Helper {}, defined elements order is not valid. It should be 'row'/'column'"
                .format(helper.user_name))

    def _parse_tiles_order(self, json: dict, helper):
        # Used for helpers only
        order = json[jd.TILES_ORDER_KEY]
        if order.lower() == 'row':
            helper.tiles_order = fblas_types.FblasOrder.FblasRowMajor
            return True
        elif order.lower() == 'column':
            helper.tiles_order = fblas_types.FblasOrder.FblasColMajor
            return True
        else:
            logging.warning(
                "Helper {}, defined tiles order is not valid. It should be 'row'/'column'"
                .format(helper.user_name))

    def _parse_stride(self, json: dict, helper):
        stride = json[jd.STRIDE_KEY]
        if not isinstance(stride, int):
            logging.warning(
                "Helper {}, defined incx is not a valid number".format(
                    helper.user_name))
            return False
        else:
            helper.stride = stride
            return True

    def _parse_tiles_A_order(self, json: dict,
                             routine: fblas_routine.FBLASRoutine):
        order = json[jd.TILES_A_ORDER_KEY]
        if order.lower() == 'row':
            routine.tiles_A_order = fblas_types.FblasOrder.FblasRowMajor
            return True
        elif order.lower() == 'column':
            routine.tiles_A_order = fblas_types.FblasOrder.FblasColMajor
            return True
        else:
            logging.warning(
                "Routine {}, defined tiles A elements order is not valid. It should be 'row'/'column'"
                .format(routine.user_name))

    def _parse_vect_size(self, json: dict,
                         routine: fblas_routine.FBLASRoutine):
        vect_size = json[jd.VECT_SIZE_KEY]
        if not isinstance(vect_size, int) or vect_size not in {1, 2, 4, 8, 16}:
            logging.warning(
                "Routine {}, defined vect_size is not a valid number (must be a positive power of 2, less than 16)."
                "Will use default vector size.".format(routine.user_name))
            return False
        else:
            routine.vect_size = vect_size
            return True

    def _parse_elements_A_order(self, json: dict,
                                routine: fblas_routine.FBLASRoutine):
        order = json[jd.ELEMENTS_A_ORDER_KEY]
        if order.lower() == 'row':
            routine.elements_A_order = fblas_types.FblasOrder.FblasRowMajor
            return True
        elif order.lower() == 'column':
            routine.elements_A_order = fblas_types.FblasOrder.FblasColMajor
            return True
        else:
            logging.warning(
                "Routine {}, defined A elements order is not valid. It should be 'row'/'column'"
                .format(routine.user_name))

    def _parse_input_channels(self, json: dict,
                              routine: fblas_routine.FBLASRoutine):
        """
        For all required input channels, take the user specified name and add it to the routine
        :return: True if all the required input channels have been successfully parsed, false otherwise
        """
        for chan in self._modules_codegen_required_input_channels[
                routine.blas_name]:
            if not self._is_json_key_present(json, chan):
                logging.warning(
                    "Routine {}, has not required input channel {}".format(
                        routine.user_name, chan))
                return False
            else:
                routine.add_input_channel(chan, json[chan])
        return True

    def _parse_output_channels(self, json: dict,
                               routine: fblas_routine.FBLASRoutine):
        """
        For all required output channels, take the user specified name and add it to the routine
        :return: True if all the required output channels have been successfully parsed, false otherwise
        """
        for chan in self._modules_codegen_required_output_channels[
                routine.blas_name]:
            if not self._is_json_key_present(json, chan):
                logging.warning(
                    "Routine {}, has not required output channel {}".format(
                        routine.user_name, chan))
                return False
            else:
                routine.add_output_channel(chan, json[chan])
        if routine.blas_name in self._modules_codegen_optional_output_channels:
            for chan in self._modules_codegen_optional_output_channels[
                    routine.blas_name]:
                if self._is_json_key_present(json, chan):
                    routine.add_output_channel(chan, json[chan])
        return True

    def _parse_routine_attribute(self, json: dict, field: str,
                                 routine: fblas_routine.FBLASRoutine,
                                 required: bool):
        """
        Parse a routine attribute from the json definition, updating the routine accordingly
        :param json: the json key containing the routine definition
        :param field: field to parse
        :param routine: the routine object
        :param required: true if the field is required
        :return: true if the field has been properly parsed, false otherwise. This modifies the routine object
        """

        # check if the routine definition has the attribute
        # and then dispatch to the proper method

        if self._is_json_key_present(json, field):
            if field.lower() == "tile n size":
                method_name = "_parse_tile_n_size"
            elif field.lower() == "tile m size":
                method_name = "_parse_tile_m_size"
            elif field.lower() == jd.TILES_ORDER_KEY:
                method_name = "_parse_tiles_order"
            elif field.lower() == jd.ELEMENTS_ORDER_KEY:
                method_name = "_parse_elements_order"
            elif field.lower() == "a tiles order":
                method_name = "_parse_tiles_A_order"
            elif field.lower() == "a elements order":
                method_name = "_parse_elements_A_order"
            elif field.lower() == jd.WIDTH_X_KEY:
                method_name = "_parse_width_x"
            elif field.lower() == jd.WIDTH_Y_KEY:
                method_name = "_parse_width_y"
            elif field.lower() == jd.TILE_SIZE_KEY:
                method_name = "_parse_tile_size"
            elif field.lower() == jd.VECT_SIZE_KEY:
                method_name = "_parse_vect_size"
            else:
                method_name = "_parse_" + field
            method = getattr(self, method_name)
            ret = method(json, routine)
            if not ret and required:
                logging.warning(
                    "Problem in parsing routine {}: required field {} missing".
                    format(routine.user_name, field))
            return ret
        else:
            if required:
                logging.warning(
                    "Problem in parsing routine {}: required field {} missing".
                    format(routine.user_name, field))
            return required

    def _parse_helper_attribute(self, json: dict, field: str,
                                helper: fblas_helper.FBLASHelper,
                                required: bool):
        '''
        Parse an helper attribute from the json definition, updating the helper accordingly
        :param json: the json key containing the routine definition
        :param field: field to parse
        :param helper: the helper object
        :param required: true if the field is required
        :return: true if the field has been properly parsed, false otherwise. This modifies the helper object
        '''

        # check if the routine definition has the attribute
        # and then dispatch to the proper method
        if self._is_json_key_present(json, field):
            if field.lower() == 'tile n size':
                method_name = "_parse_tile_n_size"
            elif field.lower() == 'tile m size':
                method_name = "_parse_tile_m_size"
            elif field.lower() == 'tiles order':
                method_name = "_parse_tiles_order"
            elif field.lower() == 'elements order':
                method_name = "_parse_elements_order"
            else:
                method_name = "_parse_" + field
            method = getattr(self, method_name)
            ret = method(json, helper)
            if not ret and required:
                logging.warning(
                    "Problem in parsing helper {}: required field {} missing".
                    format(helper.user_name, field))
            return ret
        else:
            if required:
                logging.warning(
                    "Problem in parsing helper {}: required field {} missing".
                    format(helper.user_name, field))
            return required

    def _parse_routine(self, json: dict) -> fblas_routine.FBLASRoutine:
        '''
        Parses a routine from the given json key
        :param json: the json key containing the routine definition
        :return: a routine object. Malformed routines are dropped and a warning is displayed
        '''
        # check for basic required routine elements
        # Blas_Name
        if not self._is_json_key_present(json, jd.BLAS_NAME_KEY):
            logging.warning("Found routine definition without {} key".format(
                jd.BLAS_NAME_KEY))
            return None

        blas_name = json[jd.BLAS_NAME_KEY]

        # check that blas_name is a valid/supported blas routine
        supported = (self._codegen == fblas_types.FblasCodegen.HostCodegen and blas_name in self._host_api_supported_routines) \
                    or (self._codegen == fblas_types.FblasCodegen.ModulesCodegen and blas_name in self._modules_codegen_supported_routines)
        if not supported:
            logging.warning(
                "{} is not a supported BLAS Routine".format(blas_name))
            return None

        # user_name
        if not self._is_json_key_present(json, jd.USER_NAME_KEY):
            logging.warning("Found routine definition without {} key".format(
                jd.USER_NAME_KEY))
            return None

        user_name = json[jd.USER_NAME_KEY]

        # type
        if not self._is_json_key_present(json, jd.TYPE_KEY):
            logging.warning("Found routine definition without {} key".format(
                jd.TYPE_KEY))
            return None

        type = json[jd.TYPE_KEY]

        if type not in fblas_types.TYPE_STR_TO_ROUTINE_TYPE:
            logging.warning("Type {} for routine {} not supported".format(
                type, user_name))
            return None

        # Create the routine
        routine = fblas_routine.FBLASRoutine(
            blas_name, user_name, fblas_types.TYPE_STR_TO_ROUTINE_TYPE[type],
            self._platform, self._codegen)

        # Parse required fields
        has_req_fields = True

        if self._is_host_codegen:
            if blas_name in self._host_api_required_parameters:
                for field in self._host_api_required_parameters[blas_name]:
                    has_req_fields = has_req_fields and self._parse_routine_attribute(
                        json, field, routine, True)
        else:
            if blas_name in self._modules_codegen_required_parameters:
                for field in self._modules_codegen_required_parameters[
                        blas_name]:
                    has_req_fields = has_req_fields and self._parse_routine_attribute(
                        json, field, routine, True)

        if not has_req_fields:
            return None

        if self._is_host_codegen:
            if blas_name in self._host_api_optional_parameters:
                for field in self._host_api_optional_parameters[blas_name]:
                    self._parse_routine_attribute(json, field, routine, False)
        else:
            if blas_name in self._modules_codegen_optional_parameters:
                for field in self._modules_codegen_optional_parameters[
                        blas_name]:
                    self._parse_routine_attribute(json, field, routine, False)

        # Double level of tiling: perform additional checks
        if routine.has_2D_computational_tile:
            if routine.tile_size % routine.width_x != 0 or routine.tile_size % routine.width_y != 0:
                logging.warning(
                    "Tile size {} for routine {} must be a multiple of width x and width y"
                    .format(type, user_name))
                return None

        # if we are codegen modules, we have to read also the channels
        if not self._is_host_codegen:
            has_req_channels = True
            if len(self._modules_codegen_required_input_channels[blas_name]
                   ) > 0:
                has_req_channels = has_req_channels and self._parse_input_channels(
                    json, routine)
            if len(self._modules_codegen_required_output_channels[blas_name]
                   ) > 0:
                has_req_channels = has_req_channels and self._parse_output_channels(
                    json, routine)
            if not has_req_channels:
                return None

        return routine

    def _parse_helper(self, json: dict) -> fblas_helper.FBLASHelper:
        '''
        Parses an helper from the given json key
        :param json: the key containing the helper definition
        :return: the helper representation
        '''

        # check for basic required routine elements
        # Blas_Name
        if not self._is_json_key_present(json, jd.HELPER_NAME_KEY):
            logging.warning("Found helper definition without {} key".format(
                jd.HELPER_NAME_KEY))
            return None

        helper_name = json[jd.HELPER_NAME_KEY]

        # check that blas_name is a valid/supported blas helper
        if helper_name not in self._modules_codegen_supported_helpers:
            logging.warning("{} is not a supported helper".format(helper_name))
            return None

        # user_name
        if not self._is_json_key_present(json, jd.USER_NAME_KEY):
            logging.warning("Found helper definition without {} key".format(
                jd.USER_NAME_KEY))
            return None

        user_name = json[jd.USER_NAME_KEY]

        # type
        if not self._is_json_key_present(json, jd.TYPE_KEY):
            logging.warning("Found helper definition without {} key".format(
                jd.TYPE_KEY))
            return None

        type = json[jd.TYPE_KEY]

        if type not in fblas_types.TYPE_STR_TO_ROUTINE_TYPE:
            logging.warning("Type {} for helper {} not supported".format(
                type, user_name))
            return None

        #get the name

        if not self._is_json_key_present(json, jd.HELPER_CHANNEL_NAME_KEY):
            logging.warning("Found helper definition without {} key".format(
                jd.HELPER_NAME_KEY))
            return None

        channel_name = json[jd.HELPER_CHANNEL_NAME_KEY]

        # Create the helper
        helper = fblas_helper.FBLASHelper(
            helper_name, user_name, channel_name,
            fblas_types.TYPE_STR_TO_ROUTINE_TYPE[type])

        # Parse required fields

        has_req_fields = True
        if helper_name in self._modules_codegen_helpers_required_parameters:
            for field in self._modules_codegen_helpers_required_parameters[
                    helper_name]:
                has_req_fields = has_req_fields and self._parse_helper_attribute(
                    json, field, helper, True)
        if not has_req_fields:
            return None

        if helper_name in self._modules_codegen_helpers_optional_parameters:
            for field in self._modules_codegen_helpers_optional_parameters[
                    helper_name]:
                self._parse_helper_attribute(json, field, helper, False)

        return helper

    def parse_json(self, file_path: str):

        with open(file_path) as json_file:
            document = json.load(json_file)

        # check if there is some routine definition
        if not self._is_json_key_present(document, jd.ROUTINE_KEY):
            raise RuntimeError(
                "Error: no routine definition present in the JSON file")

        # get platform if present
        if not self._is_json_key_present(document, jd.PLATFORM_KEY):
            logging.warning(
                "No platform definition present in the JSON file. {} will be used as target platform"
                .format(jd.DEFAULT_PLATFORM))

        else:
            if document[jd.PLATFORM_KEY] == jd.STRATIX10:
                self._platform = fblas_types.Platform.Stratix10
            elif document[jd.PLATFORM_KEY] == jd.ARRIA10:
                self._platform = fblas_types.Platform.Arria10
            else:
                raise RuntimeError("Error: platform {} not supported".format(
                    document[jd.PLATFORM_KEY]))

        # parse all the routine presents
        routines = []

        if isinstance(document[jd.ROUTINE_KEY], dict):
            parsed_routine = self._parse_routine(document[jd.ROUTINE_KEY])
            if parsed_routine is not None:
                routines.append(parsed_routine)
        else:  # array
            for r in document[jd.ROUTINE_KEY]:
                parsed_routine = self._parse_routine(r)
                if parsed_routine is not None:
                    routines.append(parsed_routine)

        if self._codegen == fblas_types.FblasCodegen.HostCodegen:
            return routines
        else:
            # Modules codegen, we have to parse the helpers
            helpers = []
            # check if there is some routine definition
            if self._is_json_key_present(document, jd.HELPER_KEY):
                if isinstance(document[jd.HELPER_KEY], dict):
                    parsed_helper = self._parse_helper(document[jd.HELPER_KEY])
                    if parsed_helper is not None:
                        helpers.append(parsed_helper)
                else:  # array
                    for h in document[jd.HELPER_KEY]:
                        parsed_helper = self._parse_helper(h)
                        if parsed_helper is not None:
                            helpers.append(parsed_helper)
            return routines, helpers
