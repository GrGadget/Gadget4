import re 
import sys
import os

def out1(fname):
    with open(fname, "w") as f:
        f.write('static const char *compiler_flags="";')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python config-cflags.py <build_dir>")
        exit(1)
    out1(sys.argv[1] + "/compiler-command-line-args.h")
