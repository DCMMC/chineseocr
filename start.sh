#!/bin/bash
# workaround to ArchLinux where opencv2 is conflict with opencv4
env LD_LIBRARY_PATH=/opt/opencv2/lib:${LD_LIBRARY_PATH} python app.py 8080 && rm -v ./core
