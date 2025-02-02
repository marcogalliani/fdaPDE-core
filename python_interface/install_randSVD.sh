r#!/bin/sh

 # Set defaults
 SCRIPT_NAME=$(basename "$0")
 BUILD_DIR=build/
 COMPILER="gcc"

 help()
 {
     echo "Usage: $SCRIPT_NAME [options]

        -c --compiler         Sets compiler (gcc/clang), default gcc
        -h --help             Shows this message"
     exit 2
 }

 clean_build_dir()
 {
     if [ -d "$BUILD_DIR" ]; then
         rm -rf "$BUILD_DIR"
     fi
 }

 ## Parse command line inputs
 SHORT=c:,h
 LONG=compiler:,help
 OPTS=$(getopt -a --n "$SCRIPT_NAME" --options $SHORT --longoptions $LONG -- "$@")
 eval set -- "$OPTS"

 while :; do
     case "$1" in
         -c | --compiler )
             COMPILER="$2"
             shift 2
             ;;
         -h | --help )
             help
             ;;
         --)
             shift;
             break
             ;;
         *)
             echo "Unexpected option: $1"
             help
             ;;
     esac
 done

 ## Set CMake compiler
 if [ "$COMPILER" = "gcc" ]; then
     export CC=$(which gcc)
     export CXX=$(which g++)
 elif [ "$COMPILER" = "clang" ]; then
     export CC=$(which clang)
     export CXX=$(which clang++)
 fi

 # Clean and recreate build directory
 clean_build_dir
 mkdir "$BUILD_DIR"
 cd "$BUILD_DIR" || exit 1

 # Run CMake and Make
 cmake ..
 make