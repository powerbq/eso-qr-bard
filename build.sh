#!/bin/bash

cd $(dirname $0)

mkdir -p build

cd build

pyinstaller --clean --distpath dist --workpath work ../app.spec
