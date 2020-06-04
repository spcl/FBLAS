#!/usr/bin/env python3
import os
import sys
import argparse
from codegen import fblas_types
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from codegen import fblas_codegen
from codegen import json_parser
from codegen import fblas_types
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("json_file")
    parser.add_argument("-output_dir", type=str, default="/tmp/")

    args = parser.parse_args()
    jd = json_parser.JSONParser(fblas_types.FblasCodegen.HostCodegen)
    r = jd.parse_json(args.json_file)
    codegen = fblas_codegen.FBLASCodegen(args.output_dir, fblas_types.FblasCodegen.HostCodegen)
    codegen.generateRoutines(r)


