#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
ray start --head \
--port=53769 \
--num-gpus=8 \
--dashboard-port=53768 \
--dashboard-host=0.0.0.0 \
--temp-dir="/data/chenhaotian/tmp"