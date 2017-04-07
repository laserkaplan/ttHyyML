#!/bin/bash

setupATLAS
lsetup root
source ve/bin/activate

export PATH="`pwd`:${PATH}"
export PYTHONPATH="`pwd`:${PYTHONPATH}"
