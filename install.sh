#!/bin/bash

CUR_DIR=`dirname ${BASH_SOURCE[0]}`

NB_CONFIG_DIR=$(jupyter --config-dir)
NB_DATA_DIR=$(jupyter --data-dir)

cp -aprf $CUR_DIR/jupyter/* $NB_CONFIG_DIR/
cp -aprf $CUR_DIR/local/share/jupyter/* $NB_DATA_DIR/
