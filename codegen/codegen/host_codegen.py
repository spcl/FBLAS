import json
from codegen import json_definitions as jd
from codegen import json_writer as jw
from codegen import fblas_routine
from codegen import fblas_types
import codegen.generator_definitions as gd
from codegen.fblas_helper import FBLASHelper
import logging
import os
import jinja2
from typing import List


class HostAPICodegen:

    _output_path = ""

    def __init__(self, output_path: str):
        self._output_path = output_path


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
            #dispatch
            method_name = "_codegen_" + r.blas_name
            method = getattr(self, method_name)
            jr = method(r, routine_id)
            routine_id = routine_id + 1
            json_routines.append(jr)

        #Output json for generated routines
        json_content = {"routine": json_routines}
        jw.write_to_file(self._output_path+"generated_routines.json", json_content)




    def _write_file(self, path, content, append=False):
        print("Generating file: "+path)
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

    def _codegen_dot(self, routine: fblas_routine.FBLASRoutine, id: int):
        template = self._read_template_file("1/dot.cl")
        chan_in_x_name = gd.CHANNEL_IN_VECTOR_X_BASE_NAME+str(id)
        chan_in_y_name = gd.CHANNEL_IN_VECTOR_Y_BASE_NAME+str(id)
        chan_out = gd.CHANNEL_OUT_SCALAR_BASE_NAME+str(id)
        channels_routine = {"channel_in_vector_x": chan_in_x_name, "channel_in_vector_y": chan_in_y_name, "channel_out_scalar": chan_out}
        output_path = self._output_path + "/" + routine.user_name+".cl"
        self._write_file(output_path, template.render(routine=routine, channels=channels_routine))

        #add helpers
        template = self._read_template_file("helpers/"+gd.TEMPLATE_READ_VECTOR_X)
        channels_helper = {"channel_out_vector": chan_in_x_name}
        helper_name_read_x = gd.HELPER_READ_VECTOR_X_BASE_NAME+str(id)
        self._write_file(output_path, template.render(helper_name=helper_name_read_x, helper=routine, channels=channels_helper), append=True)

        #Read y
        template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_VECTOR_Y)
        channels_helper = {"channel_out_vector": chan_in_y_name}
        helper_name_read_y = gd.HELPER_READ_VECTOR_Y_BASE_NAME + str(id)
        self._write_file(output_path, template.render(helper_name=helper_name_read_y, helper=routine, channels=channels_helper),
                                                      append=True)

        #Write scalar
        template = self._read_template_file("helpers/" + gd.TEMPLATE_WRITE_SCALAR)
        channels_helper = {"channel_in_scalar": chan_out}
        helper_name_write_scalar = gd.HELPER_WRITE_SCALAR_BASE_NAME + str(id)
        self._write_file(output_path, template.render(helper_name=helper_name_write_scalar, helper=routine, channels=channels_helper),
                                                      append=True)

        #create the json entries
        json = {}
        jw.add_commons(json, routine)
        jw.add_incx(json, routine)
        jw.add_incy(json, routine)
        jw.add_item(json, jd.GENERATED_READ_VECTOR_X, helper_name_read_x)
        jw.add_item(json, jd.GENERATED_READ_VECTOR_Y, helper_name_read_y)
        jw.add_item(json, jd.GENERATED_WRITE_SCALAR, helper_name_write_scalar)

        return json


    def _codegen_axpy(self, routine: fblas_routine.FBLASRoutine, id: int):
        template = self._read_template_file("1/axpy.cl")
        chan_in_x_name = gd.CHANNEL_IN_VECTOR_X_BASE_NAME+str(id)
        chan_in_y_name = gd.CHANNEL_IN_VECTOR_Y_BASE_NAME+str(id)
        chan_out = gd.CHANNEL_OUT_VECTOR_BASE_NAME+str(id)
        channels_routine = {"channel_in_vector_x": chan_in_x_name, "channel_in_vector_y": chan_in_y_name, "channel_out_vector": chan_out}
        output_path = self._output_path + "/" + routine.user_name+".cl"
        self._write_file(output_path, template.render(routine=routine, channels=channels_routine))

        #add helpers
        template = self._read_template_file("helpers/"+gd.TEMPLATE_READ_VECTOR_X)
        channels_helper = {"channel_out_vector": chan_in_x_name}
        helper_name_read_x = gd.HELPER_READ_VECTOR_X_BASE_NAME+str(id)
        self._write_file(output_path, template.render(helper_name=helper_name_read_x, helper=routine, channels=channels_helper), append=True)

        #Read y
        template = self._read_template_file("helpers/" + gd.TEMPLATE_READ_VECTOR_Y)
        channels_helper = {"channel_out_vector": chan_in_y_name}
        helper_name_read_y = gd.HELPER_READ_VECTOR_Y_BASE_NAME + str(id)
        self._write_file(output_path, template.render(helper_name=helper_name_read_y, helper=routine, channels=channels_helper),
                                                      append=True)

        #Write vector
        template = self._read_template_file("helpers/" + gd.TEMPLATE_WRITE_VECTOR)
        channels_helper = {"channel_in_vector": chan_out}
        helper_name_write_vector = gd.HELPER_WRITE_VECTOR_BASE_NAME + str(id)
        self._write_file(output_path, template.render(helper_name=helper_name_write_vector, helper=routine, channels=channels_helper),
                                                      append=True)

        #create the json entries
        json = {}
        jw.add_commons(json, routine)
        jw.add_incx(json, routine)
        jw.add_incy(json, routine)
        jw.add_item(json, jd.GENERATED_READ_VECTOR_X, helper_name_read_x)
        jw.add_item(json, jd.GENERATED_READ_VECTOR_Y, helper_name_read_y)
        jw.add_item(json, jd.GENERATED_WRITE_VECTOR, helper_name_write_vector)

        return json





