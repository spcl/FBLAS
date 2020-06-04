import json
from codegen import json_definitions as jd
from codegen import json_writer as jw
from codegen import fblas_routine
from codegen import fblas_types
import codegen.generator_definitions as gd
from codegen import fblas_helper
import logging
import os
import jinja2
from typing import List


class FBLASCodegen:
    _output_path = ""
    _codegen = None  # Type of codegen
    _is_host_codegen = False #True if this instance is for host codegen

    def __init__(self, output_path: str, codegen: fblas_types.FblasCodegen):
        self._output_path = output_path
        self._codegen = codegen
        self._is_host_codegen = codegen == fblas_types.FblasCodegen.HostCodegen

    def generateRoutines(self, routines: List[fblas_routine.FBLASRoutine]):
        """
        Generates the code for the given routines
        :param routines:
        :return:
        """
        routine_id = 0
        json_routines = []
        for r in routines:
            print("Generating: " + r.user_name)
            # dispatch
            method_name = "_codegen_" + r.blas_name
            method = getattr(self, method_name)
            jr = method(r, routine_id)
            routine_id = routine_id + 1
            json_routines.append(jr)

        if self._is_host_codegen:
            # Output json for generated routines
            json_content = {"routine": json_routines}
            jw.write_to_file(self._output_path + "/generated_routines.json", json_content)

    def generateHelpers(self, helpers: List[fblas_helper.FBLASHelper]):
        """
        Generates the code for the given helper
        :param helpers: list of helpers
        """
        helper_id = 0
        for h in helpers:
            print("Generating herlper: " + h.user_name)
            # dispatch
            method_name = "_codegen_helper_" + h.helper_name
            method = getattr(self, method_name)
            method(h)
            helper_id = helper_id + 1

    def _write_file(self, path, content, append=False):
        print("Generating file: " + path)
        with open(path, "a" if append else "w") as f:
            if append is True:
                f.write("\n")
            f.write(content)

    def _read_template_file(self, path):
        templates = os.path.join(os.path.dirname(__file__), "../../templates")
        loader = jinja2.FileSystemLoader(searchpath=templates)

        logging.basicConfig()
        logger = logging.getLogger('logger')
        logger = jinja2.make_logging_undefined(logger=logger, base=jinja2.Undefined)

        env = jinja2.Environment(loader=loader, undefined=logger)
        env.lstrip_blocks = True
        env.trim_blocks = True
        return env.get_template(path)

    ###############################################################################################################
    #
    # LEVEL 1 BLAS ROUTINES
    #
    ###############################################################################################################

    def _codegen_asum(self, routine: fblas_routine.FBLASRoutine, id: int):
        template = self._read_template_file("1/asum.cl")
        chan_in_x_name = gd.CHANNEL_IN_VECTOR_X_BASE_NAME + str(id)
        chan_out = gd.CHANNEL_OUT_SCALAR_BASE_NAME + str(id)
        channels_routine = {"channel_in_vector_x": chan_in_x_name, "channel_out_scalar": chan_out}
        output_path = self._output_path + "/" + routine.user_name + ".cl"
        self._write_file(output_path, template.render(routine=routine, channels=channels_routine))

        # add helpers
        # Read x
        template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_VECTOR_X)
        channels_helper = {"channel_out_vector": chan_in_x_name}
        helper_name_read_x = gd.HELPER_READ_VECTOR_X_BASE_NAME + str(id)
        self._write_file(output_path,
                         template.render(helper_name=helper_name_read_x, helper=routine, channels=channels_helper),
                         append=True)

        # Write scalar
        template = self._read_template_file("helpers/" + gd.TEMPLATE_WRITE_SCALAR)
        channels_helper = {"channel_in_scalar": chan_out}
        helper_name_write_scalar = gd.HELPER_WRITE_SCALAR_BASE_NAME + str(id)
        self._write_file(output_path, template.render(helper_name=helper_name_write_scalar, helper=routine,
                                                      channels=channels_helper),
                         append=True)

        # create the json entries
        json = {}
        jw.add_commons(json, routine)
        jw.add_incx(json, routine)
        jw.add_item(json, jd.GENERATED_READ_VECTOR_X, helper_name_read_x)
        jw.add_item(json, jd.GENERATED_WRITE_SCALAR, helper_name_write_scalar)

        return json

    def _codegen_axpy(self, routine: fblas_routine.FBLASRoutine, id: int):
        template = self._read_template_file("1/axpy.cl")

        chan_in_x_name = gd.CHANNEL_IN_VECTOR_X_BASE_NAME + str(id) if self._is_host_codegen else routine.input_channels["in_x"]
        chan_in_y_name = gd.CHANNEL_IN_VECTOR_Y_BASE_NAME + str(id) if self._is_host_codegen else routine.input_channels["in_y"]
        chan_out = gd.CHANNEL_OUT_VECTOR_BASE_NAME + str(id) if self._is_host_codegen else routine.output_channels["out_res"]

        channels_routine = {"channel_in_vector_x": chan_in_x_name, "channel_in_vector_y": chan_in_y_name,
                            "channel_out_vector": chan_out}
        output_path = self._output_path + "/" + routine.user_name + ".cl"
        self._write_file(output_path, template.render(routine=routine, channels=channels_routine))

        if self._is_host_codegen:
            # add helpers
            template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_VECTOR_X)
            channels_helper = {"channel_out_vector": chan_in_x_name}
            helper_name_read_x = gd.HELPER_READ_VECTOR_X_BASE_NAME + str(id)
            self._write_file(output_path,
                             template.render(helper_name=helper_name_read_x, helper=routine, channels=channels_helper),
                             append=True)

            # Read y
            template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_VECTOR_Y)
            channels_helper = {"channel_out_vector": chan_in_y_name}
            helper_name_read_y = gd.HELPER_READ_VECTOR_Y_BASE_NAME + str(id)
            self._write_file(output_path,
                             template.render(helper_name=helper_name_read_y, helper=routine, channels=channels_helper),
                             append=True)

            # Write vector
            template = self._read_template_file("helpers/" + gd.TEMPLATE_WRITE_VECTOR)
            channels_helper = {"channel_in_vector": chan_out}
            helper_name_write_vector = gd.HELPER_WRITE_VECTOR_BASE_NAME + str(id)
            self._write_file(output_path, template.render(helper_name=helper_name_write_vector, helper=routine,
                                                          channels=channels_helper, incw=routine.incy),
                             append=True)

            # create the json entries
            json = {}
            jw.add_commons(json, routine)
            jw.add_incx(json, routine)
            jw.add_incy(json, routine)
            jw.add_item(json, jd.GENERATED_READ_VECTOR_X, helper_name_read_x)
            jw.add_item(json, jd.GENERATED_READ_VECTOR_Y, helper_name_read_y)
            jw.add_item(json, jd.GENERATED_WRITE_VECTOR, helper_name_write_vector)
            return json

    def _codegen_copy(self, routine: fblas_routine.FBLASRoutine, id: int):
        # Copy is created by using the vector reader and writer helpers

        template = self._read_template_file("1/axpy.cl")
        chan_in_x_name = gd.CHANNEL_IN_VECTOR_X_BASE_NAME + str(id)

        output_path = self._output_path + "/" + routine.user_name + ".cl"

        # we should add channel definitions
        self._write_file(output_path, "#pragma OPENCL EXTENSION cl_intel_channels : enable")
        if routine.type is fblas_types.RoutineType.Double:
            self._write_file(output_path, "#pragma OPENCL EXTENSION cl_khr_fp64 : enable", append=True)
        self._write_file(output_path,
                         "channel {} {} __attribute__((depth({})));".format(routine.type_str, chan_in_x_name,
                                                                            routine.width), append=True)

        # add helpers
        # Read x
        template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_VECTOR_X)
        channels_helper = {"channel_out_vector": chan_in_x_name}
        helper_name_read_x = gd.HELPER_READ_VECTOR_X_BASE_NAME + str(id)
        self._write_file(output_path,
                         template.render(helper_name=helper_name_read_x, helper=routine, channels=channels_helper),
                         append=True)

        # Write vector
        template = self._read_template_file("helpers/" + gd.TEMPLATE_WRITE_VECTOR)
        channels_helper = {"channel_in_vector": chan_in_x_name}
        helper_name_write_vector = gd.HELPER_WRITE_VECTOR_BASE_NAME + str(id)
        self._write_file(output_path, template.render(helper_name=helper_name_write_vector, helper=routine,
                                                      channels=channels_helper, incw=routine.incy),
                         append=True)

        # create the json entries
        json = {}
        jw.add_commons(json, routine)
        jw.add_incx(json, routine)
        jw.add_incy(json, routine)
        jw.add_item(json, jd.GENERATED_READ_VECTOR_X, helper_name_read_x)
        jw.add_item(json, jd.GENERATED_WRITE_VECTOR, helper_name_write_vector)
        return json

    def _codegen_dot(self, routine: fblas_routine.FBLASRoutine, id: int):
        template = self._read_template_file("1/dot.cl")
        # This is a special case: we have the optimized implementation that requires shift registers
        routine.uses_shift_registers = True

        chan_in_x_name = gd.CHANNEL_IN_VECTOR_X_BASE_NAME + str(id) if self._is_host_codegen else routine.input_channels["in_x"]
        chan_in_y_name = gd.CHANNEL_IN_VECTOR_Y_BASE_NAME + str(id) if self._is_host_codegen else routine.input_channels["in_y"]
        chan_out = gd.CHANNEL_OUT_SCALAR_BASE_NAME + str(id) if self._is_host_codegen else routine.output_channels["out_res"]
        channels_routine = {"channel_in_vector_x": chan_in_x_name, "channel_in_vector_y": chan_in_y_name,
                            "channel_out_scalar": chan_out}
        output_path = self._output_path + "/" + routine.user_name + ".cl"
        self._write_file(output_path, template.render(routine=routine, channels=channels_routine))

        if self._is_host_codegen:
            # add helpers
            template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_VECTOR_X)
            channels_helper = {"channel_out_vector": chan_in_x_name}
            helper_name_read_x = gd.HELPER_READ_VECTOR_X_BASE_NAME + str(id)
            self._write_file(output_path,
                             template.render(helper_name=helper_name_read_x, helper=routine, channels=channels_helper),
                             append=True)

            # Read y
            template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_VECTOR_Y)
            channels_helper = {"channel_out_vector": chan_in_y_name}
            helper_name_read_y = gd.HELPER_READ_VECTOR_Y_BASE_NAME + str(id)
            self._write_file(output_path,
                             template.render(helper_name=helper_name_read_y, helper=routine, channels=channels_helper),
                             append=True)

            # Write scalar
            template = self._read_template_file("helpers/" + gd.TEMPLATE_WRITE_SCALAR)
            channels_helper = {"channel_in_scalar": chan_out}
            helper_name_write_scalar = gd.HELPER_WRITE_SCALAR_BASE_NAME + str(id)
            self._write_file(output_path, template.render(helper_name=helper_name_write_scalar, helper=routine,
                                                          channels=channels_helper),
                             append=True)

            # create the json entries
            json = {}
            jw.add_commons(json, routine)
            jw.add_incx(json, routine)
            jw.add_incy(json, routine)
            jw.add_item(json, jd.GENERATED_READ_VECTOR_X, helper_name_read_x)
            jw.add_item(json, jd.GENERATED_READ_VECTOR_Y, helper_name_read_y)
            jw.add_item(json, jd.GENERATED_WRITE_SCALAR, helper_name_write_scalar)

            return json


    def _codegen_iamax(self, routine: fblas_routine.FBLASRoutine, id: int):
        template = self._read_template_file("1/iamax.cl")
        chan_in_x_name = gd.CHANNEL_IN_VECTOR_X_BASE_NAME + str(id)
        chan_out = gd.CHANNEL_OUT_SCALAR_BASE_NAME + str(id)
        channels_routine = {"channel_in_vector_x": chan_in_x_name, "channel_out_scalar": chan_out}
        output_path = self._output_path + "/" + routine.user_name + ".cl"
        self._write_file(output_path, template.render(routine=routine, channels=channels_routine))

        # add helpers
        # Read x
        template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_VECTOR_X)
        channels_helper = {"channel_out_vector": chan_in_x_name}
        helper_name_read_x = gd.HELPER_READ_VECTOR_X_BASE_NAME + str(id)
        self._write_file(output_path,
                         template.render(helper_name=helper_name_read_x, helper=routine, channels=channels_helper),
                         append=True)

        # Write scalar
        template = self._read_template_file("helpers/" + gd.TEMPLATE_WRITE_SCALAR)
        # workaround to write the integer
        helper = routine
        helper._type_str = "int"
        channels_helper = {"channel_in_scalar": chan_out}
        helper_name_write_scalar = gd.HELPER_WRITE_SCALAR_BASE_NAME + str(id)
        self._write_file(output_path, template.render(helper_name=helper_name_write_scalar, helper=helper,
                                                      channels=channels_helper),
                         append=True)

        # create the json entries
        json = {}
        jw.add_commons(json, routine)
        jw.add_incx(json, routine)
        jw.add_item(json, jd.GENERATED_READ_VECTOR_X, helper_name_read_x)
        jw.add_item(json, jd.GENERATED_WRITE_SCALAR, helper_name_write_scalar)

        return json

    def _codegen_nrm2(self, routine: fblas_routine.FBLASRoutine, id: int):
        template = self._read_template_file("1/nrm2.cl")
        chan_in_x_name = gd.CHANNEL_IN_VECTOR_X_BASE_NAME + str(id)
        chan_out = gd.CHANNEL_OUT_SCALAR_BASE_NAME + str(id)
        channels_routine = {"channel_in_vector_x": chan_in_x_name, "channel_out_scalar": chan_out}
        output_path = self._output_path + "/" + routine.user_name + ".cl"
        self._write_file(output_path, template.render(routine=routine, channels=channels_routine))

        # add helpers
        # Read x
        template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_VECTOR_X)
        channels_helper = {"channel_out_vector": chan_in_x_name}
        helper_name_read_x = gd.HELPER_READ_VECTOR_X_BASE_NAME + str(id)
        self._write_file(output_path,
                         template.render(helper_name=helper_name_read_x, helper=routine, channels=channels_helper),
                         append=True)

        # Write scalar
        template = self._read_template_file("helpers/" + gd.TEMPLATE_WRITE_SCALAR)
        channels_helper = {"channel_in_scalar": chan_out}
        helper_name_write_scalar = gd.HELPER_WRITE_SCALAR_BASE_NAME + str(id)
        self._write_file(output_path, template.render(helper_name=helper_name_write_scalar, helper=routine,
                                                      channels=channels_helper),
                         append=True)

        # create the json entries
        json = {}
        jw.add_commons(json, routine)
        jw.add_incx(json, routine)
        jw.add_item(json, jd.GENERATED_READ_VECTOR_X, helper_name_read_x)
        jw.add_item(json, jd.GENERATED_WRITE_SCALAR, helper_name_write_scalar)

        return json

    def _codegen_rot(self, routine: fblas_routine.FBLASRoutine, id: int):
        template = self._read_template_file("1/rot.cl")
        chan_in_x_name = gd.CHANNEL_IN_VECTOR_X_BASE_NAME + str(id)
        chan_in_y_name = gd.CHANNEL_IN_VECTOR_Y_BASE_NAME + str(id)
        chan_out_x_name = gd.CHANNEL_OUT_VECTOR_X_BASE_NAME + str(id)
        chan_out_y_name = gd.CHANNEL_OUT_VECTOR_Y_BASE_NAME + str(id)
        channels_routine = {"channel_in_vector_x": chan_in_x_name, "channel_in_vector_y": chan_in_y_name,
                            "channel_out_vector_x": chan_out_x_name, "channel_out_vector_y": chan_out_y_name}
        output_path = self._output_path + "/" + routine.user_name + ".cl"
        self._write_file(output_path, template.render(routine=routine, channels=channels_routine))

        # add helpers
        template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_VECTOR_X)
        channels_helper = {"channel_out_vector": chan_in_x_name}
        helper_name_read_x = gd.HELPER_READ_VECTOR_X_BASE_NAME + str(id)
        self._write_file(output_path,
                         template.render(helper_name=helper_name_read_x, helper=routine, channels=channels_helper),
                         append=True)

        # Read y
        template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_VECTOR_Y)
        channels_helper = {"channel_out_vector": chan_in_y_name}
        helper_name_read_y = gd.HELPER_READ_VECTOR_Y_BASE_NAME + str(id)
        self._write_file(output_path,
                         template.render(helper_name=helper_name_read_y, helper=routine, channels=channels_helper),
                         append=True)

        # Write vector double
        template = self._read_template_file("helpers/" + gd.TEMPLATE_WRITE_VECTOR_X_Y)
        channels_helper = {"channel_in_vector_x": chan_out_x_name, "channel_in_vector_y": chan_out_y_name}
        helper_name_write_vector = gd.HELPER_WRITE_VECTOR_X_Y_BASE_NAME + str(id)
        self._write_file(output_path, template.render(helper_name=helper_name_write_vector, helper=routine,
                                                      channels=channels_helper),
                         append=True)

        # create the json entries
        json = {}
        jw.add_commons(json, routine)
        jw.add_incx(json, routine)
        jw.add_incy(json, routine)
        jw.add_item(json, jd.GENERATED_READ_VECTOR_X, helper_name_read_x)
        jw.add_item(json, jd.GENERATED_READ_VECTOR_Y, helper_name_read_y)
        jw.add_item(json, jd.GENERATED_WRITE_VECTOR, helper_name_write_vector)
        return json

    def _codegen_rotm(self, routine: fblas_routine.FBLASRoutine, id: int):
        template = self._read_template_file("1/rotm.cl")
        chan_in_x_name = gd.CHANNEL_IN_VECTOR_X_BASE_NAME + str(id)
        chan_in_y_name = gd.CHANNEL_IN_VECTOR_Y_BASE_NAME + str(id)
        chan_out_x_name = gd.CHANNEL_OUT_VECTOR_X_BASE_NAME + str(id)
        chan_out_y_name = gd.CHANNEL_OUT_VECTOR_Y_BASE_NAME + str(id)
        channels_routine = {"channel_in_vector_x": chan_in_x_name, "channel_in_vector_y": chan_in_y_name,
                            "channel_out_vector_x": chan_out_x_name, "channel_out_vector_y": chan_out_y_name}
        output_path = self._output_path + "/" + routine.user_name + ".cl"
        self._write_file(output_path, template.render(routine=routine, channels=channels_routine))

        # add helpers
        template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_VECTOR_X)
        channels_helper = {"channel_out_vector": chan_in_x_name}
        helper_name_read_x = gd.HELPER_READ_VECTOR_X_BASE_NAME + str(id)
        self._write_file(output_path,
                         template.render(helper_name=helper_name_read_x, helper=routine, channels=channels_helper),
                         append=True)

        # Read y
        template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_VECTOR_Y)
        channels_helper = {"channel_out_vector": chan_in_y_name}
        helper_name_read_y = gd.HELPER_READ_VECTOR_Y_BASE_NAME + str(id)
        self._write_file(output_path,
                         template.render(helper_name=helper_name_read_y, helper=routine, channels=channels_helper),
                         append=True)

        # Write vector double
        template = self._read_template_file("helpers/" + gd.TEMPLATE_WRITE_VECTOR_X_Y)
        channels_helper = {"channel_in_vector_x": chan_out_x_name, "channel_in_vector_y": chan_out_y_name}
        helper_name_write_vector = gd.HELPER_WRITE_VECTOR_X_Y_BASE_NAME + str(id)
        self._write_file(output_path, template.render(helper_name=helper_name_write_vector, helper=routine,
                                                      channels=channels_helper),
                         append=True)

        # create the json entries
        json = {}
        jw.add_commons(json, routine)
        jw.add_incx(json, routine)
        jw.add_incy(json, routine)
        jw.add_item(json, jd.GENERATED_READ_VECTOR_X, helper_name_read_x)
        jw.add_item(json, jd.GENERATED_READ_VECTOR_Y, helper_name_read_y)
        jw.add_item(json, jd.GENERATED_WRITE_VECTOR, helper_name_write_vector)
        return json

    def _codegen_rotg(self, routine: fblas_routine.FBLASRoutine, id: int):
        # For the moment being rotg is realized directly into the API (it's not worth calling an opencl kernel)
        # Therefore the output of the generation will be just an empty file
        # This has been done for compatibility with respect to the rest of the library and to leave space for further modifications

        template = self._read_template_file("helpers/empty.cl")
        output_path = self._output_path + "/" + routine.user_name + ".cl"
        self._write_file(output_path, template.render(routine=routine))

        json = {}
        jw.add_commons(json, routine)
        return json

    def _codegen_rotmg(self, routine: fblas_routine.FBLASRoutine, id: int):
        # For the moment being rotgm is realized directly into the API (it's not worth calling an opencl kernel)
        # Therefore the output of the generation will be just an empty file
        # This has been done for compatibility with respect to the rest of the library and to leave space for further modifications

        template = self._read_template_file("helpers/empty.cl")
        output_path = self._output_path + "/" + routine.user_name + ".cl"
        self._write_file(output_path, template.render(routine=routine))

        json = {}
        jw.add_commons(json, routine)
        return json

    def _codegen_scal(self, routine: fblas_routine.FBLASRoutine, id: int):
        template = self._read_template_file("1/scal.cl")
        chan_in_x_name = gd.CHANNEL_IN_VECTOR_X_BASE_NAME + str(id)
        chan_out = gd.CHANNEL_OUT_VECTOR_BASE_NAME + str(id)
        channels_routine = {"channel_in_vector_x": chan_in_x_name, "channel_out_vector": chan_out}
        output_path = self._output_path + "/" + routine.user_name + ".cl"
        self._write_file(output_path, template.render(routine=routine, channels=channels_routine))

        # Add helpers
        # Read x
        template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_VECTOR_X)
        channels_helper = {"channel_out_vector": chan_in_x_name}
        helper_name_read_x = gd.HELPER_READ_VECTOR_X_BASE_NAME + str(id)
        self._write_file(output_path,
                         template.render(helper_name=helper_name_read_x, helper=routine, channels=channels_helper),
                         append=True)

        # Write vector
        template = self._read_template_file("helpers/" + gd.TEMPLATE_WRITE_VECTOR)
        channels_helper = {"channel_in_vector": chan_out}
        helper_name_write_vector = gd.HELPER_WRITE_VECTOR_BASE_NAME + str(id)
        self._write_file(output_path, template.render(helper_name=helper_name_write_vector, helper=routine,
                                                      channels=channels_helper, incw=routine.incx),
                         append=True)

        # create the json entries
        json = {}
        jw.add_commons(json, routine)
        jw.add_incx(json, routine)
        jw.add_item(json, jd.GENERATED_READ_VECTOR_X, helper_name_read_x)
        jw.add_item(json, jd.GENERATED_WRITE_VECTOR, helper_name_write_vector)
        return json

    def _codegen_swap(self, routine: fblas_routine.FBLASRoutine, id: int):
        template = self._read_template_file("1/swap.cl")
        chan_in_x_name = gd.CHANNEL_IN_VECTOR_X_BASE_NAME + str(id)
        chan_in_y_name = gd.CHANNEL_IN_VECTOR_Y_BASE_NAME + str(id)
        chan_out_x_name = gd.CHANNEL_OUT_VECTOR_X_BASE_NAME + str(id)
        chan_out_y_name = gd.CHANNEL_OUT_VECTOR_Y_BASE_NAME + str(id)
        channels_routine = {"channel_in_vector_x": chan_in_x_name, "channel_in_vector_y": chan_in_y_name,
                            "channel_out_vector_x": chan_out_x_name, "channel_out_vector_y": chan_out_y_name}
        output_path = self._output_path + "/" + routine.user_name + ".cl"
        self._write_file(output_path, template.render(routine=routine, channels=channels_routine))

        # add helpers
        template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_VECTOR_X)
        channels_helper = {"channel_out_vector": chan_in_x_name}
        helper_name_read_x = gd.HELPER_READ_VECTOR_X_BASE_NAME + str(id)
        self._write_file(output_path,
                         template.render(helper_name=helper_name_read_x, helper=routine, channels=channels_helper),
                         append=True)

        # Read y
        template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_VECTOR_Y)
        channels_helper = {"channel_out_vector": chan_in_y_name}
        helper_name_read_y = gd.HELPER_READ_VECTOR_Y_BASE_NAME + str(id)
        self._write_file(output_path,
                         template.render(helper_name=helper_name_read_y, helper=routine, channels=channels_helper),
                         append=True)

        # Write vector double
        template = self._read_template_file("helpers/" + gd.TEMPLATE_WRITE_VECTOR_X_Y)
        channels_helper = {"channel_in_vector_x": chan_out_x_name, "channel_in_vector_y": chan_out_y_name}
        helper_name_write_vector = gd.HELPER_WRITE_VECTOR_X_Y_BASE_NAME + str(id)
        self._write_file(output_path, template.render(helper_name=helper_name_write_vector, helper=routine,
                                                      channels=channels_helper),
                         append=True)

        # create the json entries
        json = {}
        jw.add_commons(json, routine)
        jw.add_incx(json, routine)
        jw.add_incy(json, routine)
        jw.add_item(json, jd.GENERATED_READ_VECTOR_X, helper_name_read_x)
        jw.add_item(json, jd.GENERATED_READ_VECTOR_Y, helper_name_read_y)
        jw.add_item(json, jd.GENERATED_WRITE_VECTOR, helper_name_write_vector)
        return json

    ###############################################################################################################
    #
    # LEVEL 2 BLAS ROUTINES
    #
    ###############################################################################################################

    def _codegen_gemv(self, routine: fblas_routine.FBLASRoutine, id: int):

        requires_additional_channel = False
        if self._is_host_codegen:
            # Currently supported case
            if routine.order is fblas_types.FblasOrder.FblasRowMajor and routine.transposedA is fblas_types.FblasTranspose.FblasNoTrans:
                template = self._read_template_file("2/gemv_v1.cl")
                # This is a special case: we have the optimized implementation that requires shift registers
                routine.uses_shift_registers = True
            elif routine.order is fblas_types.FblasOrder.FblasRowMajor and routine.transposedA is fblas_types.FblasTranspose.FblasTrans:
                template = self._read_template_file("2/gemv_v2.cl")
            else:
                raise RuntimeError("Requirements for user routine {} are currently not supported".format(routine.user_name))
        else:
            if (routine.are_tiles_A_rowstreamed() and routine.are_elements_A_rowstreamed() and routine.transposedA is \
                fblas_types.FblasTranspose.FblasNoTrans) or (not routine.are_tiles_A_rowstreamed() and not routine.are_elements_A_rowstreamed() \
                                                             and routine.transposedA is fblas_types.FblasTranspose.FblasTrans):
                # Tiles row streamed, Elements  Rowstreamed and No Transposed
                # or Tiles col streamed, Elements ColStreamed and Transposed
                template = self._read_template_file("2/gemv_v1.cl")
                # This is a special case: we have the optimized implementation that requires shift registers
                routine.uses_shift_registers = True

            elif ( not routine.are_tiles_A_rowstreamed() and routine.are_elements_A_rowstreamed() and routine.transposedA is\
                    fblas_types.FblasTranspose.FblasTrans) or (\
                    routine.are_tiles_A_rowstreamed() and not routine.are_elements_A_rowstreamed() and \
                    routine.transposedA is fblas_types.FblasTranspose.FblasNoTrans):

                # Tiles Col Streamed,Element Row Streamed and Transposed
                # or Tiles Row Streamed, Eleemnts Col streamd and No Transp
                template = self._read_template_file("2/gemv_v2.cl")
            elif (routine.are_elements_A_rowstreamed() and routine.are_tiles_A_rowstreamed() and routine.transposedA is\
                    fblas_types.FblasTranspose.FblasTrans) or (not routine.are_elements_A_rowstreamed() and\
                    not routine.are_tiles_A_rowstreamed() and routine.transposedA is fblas_types.FblasTranspose.FblasNoTrans):
                # Tiles RowStreamed, Elements Row Streamed and Transposed
                # or Tiles Col stramed , Elements Col Streamed and No Transposed
                template = self._read_template_file("2/gemv_v4.cl")
                requires_additional_channel = True
            else:
                raise RuntimeError(
                    "Requirements for user routine {} are currently not supported".format(routine.user_name))

        chan_in_x_name = gd.CHANNEL_IN_VECTOR_X_BASE_NAME + str(id) if self._is_host_codegen else routine.input_channels["in_x"]
        chan_in_y_name = gd.CHANNEL_IN_VECTOR_Y_BASE_NAME + str(id) if self._is_host_codegen else routine.input_channels["in_y"]
        chan_in_A_name = gd.CHANNEL_IN_MATRIX_A_BASE_NAME + str(id) if self._is_host_codegen else routine.input_channels["in_A"]
        chan_out = gd.CHANNEL_OUT_VECTOR_BASE_NAME + str(id) if self._is_host_codegen else routine.output_channels["out_res"]

        channels_routine = {"channel_in_vector_x": chan_in_x_name, "channel_in_vector_y": chan_in_y_name,
                            "channel_in_matrix_A": chan_in_A_name, "channel_out_vector": chan_out}
        if requires_additional_channel: # Special case
            channels_routine.update({"channel_out_vector_y_updates" : routine.output_channels["out_y_updates"]})
        output_path = self._output_path + "/" + routine.user_name + ".cl"
        self._write_file(output_path, template.render(routine=routine, channels=channels_routine))

        if self._is_host_codegen:
            # Add helpers
            # Read x
            template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_VECTOR_X)
            channels_helper = {"channel_out_vector": chan_in_x_name}
            helper_name_read_x = gd.HELPER_READ_VECTOR_X_BASE_NAME + str(id)
            self._write_file(output_path,
                             template.render(helper_name=helper_name_read_x, helper=routine, channels=channels_helper),
                             append=True)

            # Read Y
            template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_VECTOR_Y)
            channels_helper = {"channel_out_vector": chan_in_y_name}
            helper_name_read_y = gd.HELPER_READ_VECTOR_Y_BASE_NAME + str(id)
            self._write_file(output_path,
                             template.render(helper_name=helper_name_read_y, helper=routine, channels=channels_helper),
                             append=True)

            # Read Matrix A
            if routine.order is fblas_types.FblasOrder.FblasRowMajor and routine.transposedA is fblas_types.FblasTranspose.FblasNoTrans:
                template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_MATRIX_ROWSTREAMED_TILE_ROW)
            elif routine.order is fblas_types.FblasOrder.FblasRowMajor and routine.transposedA is fblas_types.FblasTranspose.FblasTrans:
                template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_MATRIX_ROWSTREAMED_TILE_COL)
            channels_helper = {"channel_out_matrix": chan_in_A_name}
            helper_name_read_A = gd.HELPER_READ_MATRIX_A_BASE_NAME + str(id)
            self._write_file(output_path,
                             template.render(helper_name=helper_name_read_A, helper=routine, channels=channels_helper),
                             append=True)

            # Write vector
            template = self._read_template_file("helpers/" + gd.TEMPLATE_WRITE_VECTOR)
            channels_helper = {"channel_in_vector": chan_out}
            helper_name_write_vector = gd.HELPER_WRITE_VECTOR_BASE_NAME + str(id)
            self._write_file(output_path, template.render(helper_name=helper_name_write_vector, helper=routine,
                                                          channels=channels_helper, incw=routine.incy),
                             append=True)

            # create the json entries
            json = {}
            jw.add_commons(json, routine)
            jw.add_incx(json, routine)
            jw.add_incy(json, routine)
            jw.add_tile_n_size(json, routine)
            jw.add_tile_m_size(json, routine)
            jw.add_transposed(json, routine)
            jw.add_order(json, routine)
            jw.add_item(json, jd.GENERATED_READ_VECTOR_X, helper_name_read_x)
            jw.add_item(json, jd.GENERATED_READ_VECTOR_Y, helper_name_read_y)
            jw.add_item(json, jd.GENERATED_READ_MATRIX_A, helper_name_read_A)
            jw.add_item(json, jd.GENERATED_WRITE_VECTOR, helper_name_write_vector)
            return json

    def _codegen_ger(self, routine: fblas_routine.FBLASRoutine, id: int):

        # Currently supported case: in Host API only Row Major, in modules every
        if self._is_host_codegen:
            if routine.order is fblas_types.FblasOrder.FblasRowMajor:
                template = self._read_template_file("2/ger_v1.cl")
            else:
                raise RuntimeError("Requirements for user routine {} are currently not supported in Host API codegen".format(routine.user_name))
        else:
            if routine.are_tiles_A_rowstreamed():
                if routine.are_elements_A_rowstreamed():
                    template = self._read_template_file("2/ger_v1.cl")
                else:
                    template = self._read_template_file("2/ger_v4.cl")
            else:
                if routine.are_elements_A_rowstreamed():
                    template = self._read_template_file("2/ger_v3.cl")
                else:
                    emplate = self._read_template_file("2/ger_v2.cl")

        chan_in_x_name = gd.CHANNEL_IN_VECTOR_X_BASE_NAME + str(id) if self._is_host_codegen else \
        routine.input_channels["in_x"]
        chan_in_y_name = gd.CHANNEL_IN_VECTOR_Y_BASE_NAME + str(id) if self._is_host_codegen else \
        routine.input_channels["in_y"]
        chan_in_A_name = gd.CHANNEL_IN_MATRIX_A_BASE_NAME + str(id) if self._is_host_codegen else \
        routine.input_channels["in_A"]
        chan_out = gd.CHANNEL_OUT_MATRIX_BASE_NAME + str(id) if self._is_host_codegen else routine.output_channels[
            "out_res"]

        channels_routine = {"channel_in_vector_x": chan_in_x_name, "channel_in_vector_y": chan_in_y_name,
                            "channel_in_matrix_A": chan_in_A_name, "channel_out_matrix": chan_out}
        output_path = self._output_path + "/" + routine.user_name + ".cl"
        self._write_file(output_path, template.render(routine=routine, channels=channels_routine))

        if self._is_host_codegen:
            # Add helpers
            # Read x
            template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_VECTOR_X)
            channels_helper = {"channel_out_vector": chan_in_x_name}
            helper_name_read_x = gd.HELPER_READ_VECTOR_X_BASE_NAME + str(id)
            self._write_file(output_path,
                             template.render(helper_name=helper_name_read_x, helper=routine, channels=channels_helper),
                             append=True)

            # Read Y
            template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_VECTOR_Y)
            channels_helper = {"channel_out_vector": chan_in_y_name}
            helper_name_read_y = gd.HELPER_READ_VECTOR_Y_BASE_NAME + str(id)
            self._write_file(output_path,
                             template.render(helper_name=helper_name_read_y, helper=routine, channels=channels_helper),
                             append=True)

            # Read Matrix A
            if routine.order is fblas_types.FblasOrder.FblasRowMajor:
                template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_MATRIX_ROWSTREAMED_TILE_ROW)
                channels_helper = {"channel_out_matrix": chan_in_A_name}
                helper_name_read_A = gd.HELPER_READ_MATRIX_A_BASE_NAME + str(id)
                self._write_file(output_path,
                                 template.render(helper_name=helper_name_read_A, helper=routine, channels=channels_helper),
                                 append=True)
            # Write Matrix A
            if routine.order is fblas_types.FblasOrder.FblasRowMajor:
                template = self._read_template_file("helpers/" + gd.TEMPLATE_WRITE_MATRIX_ROWSTREAMED_TILE_ROW)
                channels_helper = {"channel_in_matrix": chan_out}
                helper_name_write_A = gd.HELPER_WRITE_MATRIX_BASE_NAME + str(id)
                self._write_file(output_path,
                                 template.render(helper_name=helper_name_write_A, helper=routine, channels=channels_helper),
                                 append=True)

            # create the json entries
            json = {}
            jw.add_commons(json, routine)
            jw.add_incx(json, routine)
            jw.add_incy(json, routine)
            jw.add_tile_n_size(json, routine)
            jw.add_tile_m_size(json, routine)
            jw.add_order(json, routine)
            jw.add_item(json, jd.GENERATED_READ_VECTOR_X, helper_name_read_x)
            jw.add_item(json, jd.GENERATED_READ_VECTOR_Y, helper_name_read_y)
            jw.add_item(json, jd.GENERATED_READ_MATRIX_A, helper_name_read_A)
            jw.add_item(json, jd.GENERATED_WRITE_MATRIX, helper_name_write_A)
            return json

    def _codegen_syr(self, routine: fblas_routine.FBLASRoutine, id: int):

        # Currently supported case

        if routine.order is fblas_types.FblasOrder.FblasRowMajor and routine.uplo is fblas_types.FblasUpLo.FblasLower:
            template = self._read_template_file("2/syr_v1.cl")
        elif routine.order is fblas_types.FblasOrder.FblasRowMajor and routine.uplo is fblas_types.FblasUpLo.FblasUpper:
            template = self._read_template_file("2/syr_v2.cl")
        else:
            raise RuntimeError("Requirements for user routine {} are currently not supported".format(routine.user_name))

        chan_in_x_name = gd.CHANNEL_IN_VECTOR_X_BASE_NAME + str(id)
        chan_in_x_trans_name = gd.CHANNEL_IN_VECTOR_X_TRANS_BASE_NAME + str(id)
        chan_in_A_name = gd.CHANNEL_IN_MATRIX_A_BASE_NAME + str(id)
        chan_out = gd.CHANNEL_OUT_MATRIX_BASE_NAME + str(id)
        channels_routine = {"channel_in_vector_x": chan_in_x_name, "channel_in_vector_x_trans": chan_in_x_trans_name,
                            "channel_in_matrix_A": chan_in_A_name, "channel_out_matrix": chan_out}
        output_path = self._output_path + "/" + routine.user_name + ".cl"
        self._write_file(output_path, template.render(routine=routine, channels=channels_routine))

        # Add helpers
        # Read x
        template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_VECTOR_X)
        channels_helper = {"channel_out_vector": chan_in_x_name}
        helper_name_read_x = gd.HELPER_READ_VECTOR_X_BASE_NAME + str(id)
        self._write_file(output_path,
                         template.render(helper_name=helper_name_read_x, helper=routine, channels=channels_helper),
                         append=True)

        # Read X trans
        if routine.uplo is fblas_types.FblasUpLo.FblasLower:
            template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_VECTOR_X_TRANS_LOWER)
        else:
            template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_VECTOR_X_TRANS_UPPER)
        channels_helper = {"channel_out_vector": chan_in_x_trans_name}
        helper_name_read_x_trans = gd.HELPER_READ_VECTOR_X_TRANS_BASE_NAME + str(id)
        self._write_file(output_path,
                         template.render(helper_name=helper_name_read_x_trans, helper=routine, channels=channels_helper),
                         append=True)
        # TODO: create templates for matrix/read write and adjust this
        # Read Matrix A
        if routine.order is fblas_types.FblasOrder.FblasRowMajor and routine.transposedA is fblas_types.FblasTranspose.FblasNoTrans:
            template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_MATRIX_ROWSTREAMED_TILE_ROW)
        elif routine.order is fblas_types.FblasOrder.FblasRowMajor and routine.transposedA is fblas_types.FblasTranspose.FblasTrans:
            template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_MATRIX_ROWSTREAMED_TILE_COL)
        channels_helper = {"channel_out_matrix": chan_in_A_name}
        helper_name_read_A = gd.HELPER_READ_MATRIX_A_BASE_NAME + str(id)
        self._write_file(output_path,
                         template.render(helper_name=helper_name_read_A, helper=routine, channels=channels_helper),
                         append=True)

        # Write vector
        template = self._read_template_file("helpers/" + gd.TEMPLATE_WRITE_VECTOR)
        channels_helper = {"channel_in_vector": chan_out}
        helper_name_write_vector = gd.HELPER_WRITE_VECTOR_BASE_NAME + str(id)
        self._write_file(output_path, template.render(helper_name=helper_name_write_vector, helper=routine,
                                                      channels=channels_helper, incw=routine.incy),
                         append=True)

        # create the json entries
        json = {}
        jw.add_commons(json, routine)
        jw.add_incx(json, routine)
        jw.add_incy(json, routine)
        jw.add_tile_n_size(json, routine)
        jw.add_tile_m_size(json, routine)
        jw.add_transposed(json, routine)
        jw.add_order(json, routine)
        jw.add_item(json, jd.GENERATED_READ_VECTOR_X, helper_name_read_x)
        jw.add_item(json, jd.GENERATED_READ_VECTOR_Y, helper_name_read_y)
        jw.add_item(json, jd.GENERATED_READ_MATRIX_A, helper_name_read_A)
        jw.add_item(json, jd.GENERATED_WRITE_VECTOR, helper_name_write_vector)
        return json

    ##############################################################################################
    #
    # Level 3 routines
    #
    ##############################################################################################

    def _codegen_gemm(self, routine: fblas_routine.FBLASRoutine, id: int):

        # Currently supported case: in Host API only Row Major, in modules every
        if self._is_host_codegen:
            if routine.systolic:
                template = self._read_template_file("3/gemm_systolic.cl")
            else:
                template = self._read_template_file("3/gemm.cl")
        else:
            raise RuntimeError(
                "Requirements for user routine {} are currently not supported in Host API codegen".format(
                    routine.user_name))

        chan_in_A_name = gd.CHANNEL_IN_MATRIX_A_BASE_NAME + str(id)
        chan_in_B_name = gd.CHANNEL_IN_MATRIX_B_BASE_NAME + str(id)
        chan_out_name = gd.CHANNEL_OUT_MATRIX_BASE_NAME + str(id)

        channels_routine = {"channel_in_matrix_A": chan_in_A_name, "channel_in_matrix_B": chan_in_B_name, "channel_out_matrix": chan_out_name}
        output_path = self._output_path + "/" + routine.user_name + ".cl"

        #adjust parameters for systolic
        if routine.systolic:
            #the width_x must be adjusted considering the vector size. If it is not a multiple, raise error
            if routine.width_x % routine.vect_size != 0:
                raise RuntimeError(
                    "User routine {}: width x must be a multiple of vector size".format(
                        routine.user_name))
            routine.width_x = int(routine.width_x/routine.vect_size)


        self._write_file(output_path, template.render(routine=routine, channels=channels_routine))

        if self._is_host_codegen:
            # Add helpers
            # Read A

            if routine.transposedA is fblas_types.FblasTranspose.FblasNoTrans:
                if routine.systolic:
                    template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_MATRIX_A_GEMM_NOTRANS_SYSTOLIC)
                else:
                    template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_MATRIX_A_GEMM_NOTRANS)
            else:
                if routine.systolic:
                    raise RuntimeError(
                        "User routine {}: this combination is not currently supported in systolic. Contact us!".format(
                            routine.user_name))
                else:
                    template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_MATRIX_A_GEMM_TRANS)
            channels_helper = {"channel_out_matrix": chan_in_A_name}
            helper_name_read_A = gd.HELPER_READ_MATRIX_A_BASE_NAME + str(id)
            self._write_file(output_path,
                             template.render(helper_name=helper_name_read_A, helper=routine, channels=channels_helper),
                             append=True)

            # Read B
            if routine.transposedB is fblas_types.FblasTranspose.FblasNoTrans:
                if routine.systolic:
                    template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_MATRIX_B_GEMM_NOTRANS_SYSTOLIC)
                else:
                    template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_MATRIX_B_GEMM_NOTRANS)
            else:
                if routine.systolic:
                    raise RuntimeError(
                        "User routine {}: this combination is not currently supported in systolic. Contact us!".format(
                            routine.user_name))
                else:
                    template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_MATRIX_B_GEMM_TRANS)
            channels_helper = {"channel_out_matrix": chan_in_B_name}
            helper_name_read_B = gd.HELPER_READ_MATRIX_B_BASE_NAME + str(id)
            self._write_file(output_path,
                             template.render(helper_name=helper_name_read_B, helper=routine, channels=channels_helper),
                             append=True)

            # Write gemm
            if routine.systolic:
                template = self._read_template_file("helpers/" + gd.TEMPLATE_WRITE_MATRIX_GEMM_SYSTOLIC)
            else:
                template = self._read_template_file("helpers/" + gd.TEMPLATE_WRITE_MATRIX_GEMM)
            channels_helper = {"channel_in_matrix": chan_out_name}
            helper_name_write = gd.HELPER_WRITE_MATRIX_BASE_NAME + str(id)
            self._write_file(output_path,
                             template.render(helper_name=helper_name_write, helper=routine, channels=channels_helper),
                             append=True)

            # create the json entries
            json = {}
            jw.add_commons(json, routine)
            jw.add_tile_size(json, routine)
            jw.add_transposed(json,routine)
            jw.add_transposedB(json,routine)
            jw.add_item(json, jd.GENERATED_READ_MATRIX_A, helper_name_read_A)
            jw.add_item(json, jd.GENERATED_READ_MATRIX_B, helper_name_read_B)
            jw.add_item(json, jd.GENERATED_WRITE_MATRIX, helper_name_write)
            json[jd.GENERATED_SYSTOLIC] = bool(routine.systolic)
            return json



    ##############################################################################################
    #
    # Helpers codegen
    #
    ##############################################################################################

    def _codegen_helper_read_vector_x(self, helper: fblas_helper.FBLASHelper):
        output_path = self._output_path + "/" + helper.user_name + ".cl"
        template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_VECTOR_X)
        print("Generating read vector x with widht: " + str(helper.width))
        channels_helper = {"channel_out_vector": helper.channel_name}
        self._write_file(output_path,
                         template.render(helper_name=helper.user_name, helper=helper, channels=channels_helper, generate_channel_declaration=True))

    def _codegen_helper_read_vector_y(self, helper: fblas_helper.FBLASHelper):
        output_path = self._output_path + "/" + helper.user_name + ".cl"
        template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_VECTOR_Y)
        channels_helper = {"channel_out_vector": helper.channel_name}
        self._write_file(output_path,
                         template.render(helper_name=helper.user_name, helper=helper, channels=channels_helper, generate_channel_declaration=True))

    def _codegen_helper_write_scalar(self, helper: fblas_helper.FBLASHelper):
        output_path = self._output_path + "/" + helper.user_name + ".cl"
        template = self._read_template_file("helpers/" + gd.TEMPLATE_WRITE_SCALAR)
        channels_helper = {"channel_in_scalar": helper.channel_name}
        self._write_file(output_path,
                         template.render(helper_name=helper.user_name, helper=helper, channels=channels_helper, generate_channel_declaration=True))

    def _codegen_helper_write_vector(self, helper: fblas_helper.FBLASHelper):
        output_path = self._output_path + "/" + helper.user_name + ".cl"
        template = self._read_template_file("helpers/" + gd.TEMPLATE_WRITE_VECTOR)
        channels_helper = {"channel_in_vector": helper.channel_name}
        self._write_file(output_path,
                         template.render(helper_name=helper.user_name, helper=helper, channels=channels_helper, incw=helper.incy, generate_channel_declaration=True))

    def _codegen_helper_generate_dummy_vector(self, helper: fblas_helper.FBLASHelper):
        output_path = self._output_path + "/" + helper.user_name + ".cl"
        template = self._read_template_file("helpers/" + gd.TEMPLATE_GENERATE_DUMMY_VECTOR)
        channels_helper = {"channel_out_vector": helper.channel_name}
        self._write_file(output_path,
                         template.render(helper_name=helper.user_name, helper=helper, channels=channels_helper, generate_channel_declaration=True))

    def _codegen_helper_vector_sink(self, helper: fblas_helper.FBLASHelper):
        output_path = self._output_path + "/" + helper.user_name + ".cl"
        template = self._read_template_file("helpers/" + gd.TEMPLATE_VECTOR_SINK)
        channels_helper = {"channel_in_vector": helper.channel_name}
        self._write_file(output_path,
                         template.render(helper_name=helper.user_name, helper=helper, channels=channels_helper, generate_channel_declaration=True))

    def _codegen_helper_read_matrix(self, helper: fblas_helper.FBLASHelper):
        output_path = self._output_path + "/" + helper.user_name + ".cl"
        if helper.elements_order is fblas_types.FblasOrder.FblasRowMajor:
            if helper.tiles_order is fblas_types.FblasOrder.FblasRowMajor:
                template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_MATRIX_ROWSTREAMED_TILE_ROW)
            else:
                template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_MATRIX_ROWSTREAMED_TILE_COL)
        else:
            raise RuntimeError("Requirements for user helper {} are currently not supported (currently only row streamed elements are supported)".format(helper.user_name))

        channels_helper = {"channel_out_matrix": helper.channel_name}
        self._write_file(output_path,
                         template.render(helper_name=helper.user_name, helper=helper, channels=channels_helper, generate_channel_declaration=True))

    def _codegen_helper_write_matrix(self, helper: fblas_helper.FBLASHelper):
        output_path = self._output_path + "/" + helper.user_name + ".cl"
        if helper.elements_order is fblas_types.FblasOrder.FblasRowMajor:
            if helper.tiles_order is fblas_types.FblasOrder.FblasRowMajor:
                template = self._read_template_file("helpers/" + gd.TEMPLATE_WRITE_MATRIX_ROWSTREAMED_TILE_ROW)
            else:
                #TODO
                raise RuntimeError(
                    "Requirements for user helper {} are currently not supported (currently only row streamed elements are supported)".format(
                        helper.user_name))
        else:
            raise RuntimeError(
                "Requirements for user helper {} are currently not supported (currently only row streamed elements are supported)".format(
                    helper.user_name))


        channels_helper = {"channel_in_matrix": helper.channel_name}
        self._write_file(output_path,
                         template.render(helper_name=helper.user_name, helper=helper, channels=channels_helper,
                                         generate_channel_declaration=True))

    def _codegen_helper_generate_dummy_matrix(self, helper: fblas_helper.FBLASHelper):
        output_path = self._output_path + "/" + helper.user_name + ".cl"
        template = self._read_template_file("helpers/" + gd.TEMPLATE_GENERATE_DUMMY_MATRIX)
        channels_helper = {"channel_out_matrix": helper.channel_name}
        self._write_file(output_path,
                         template.render(helper_name=helper.user_name, helper=helper, channels=channels_helper, generate_channel_declaration=True))


    def _codegen_helper_read_matrix_4_modules(self, helper: fblas_helper.FBLASHelper):
        output_path = self._output_path + "/" + helper.user_name + ".cl"
        if helper.elements_order is fblas_types.FblasOrder.FblasRowMajor and helper.tiles_order is fblas_types.FblasOrder.FblasRowMajor:
                template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_MATRIX_ROWSTREAMED_TILE_ROW_4_MODULES)
        else:
            raise RuntimeError("Requirements for user helper {} are currently not supported (currently only row streamed elements are supported)".format(helper.user_name))

        channels_helper = {"channel_out_matrix": helper.channel_name}
        self._write_file(output_path,
                         template.render(helper_name=helper.user_name, helper=helper, channels=channels_helper, generate_channel_declaration=True))