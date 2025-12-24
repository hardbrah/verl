#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES="6,7"
ray start --head \
--port=53769 \
--num-gpus=2 \
--dashboard-port=53768 \
--dashboard-host=0.0.0.0 \
--temp-dir="/mnt/nas/chenhaotian/ray_temp"