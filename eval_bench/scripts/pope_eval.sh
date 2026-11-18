#!/bin/bash

seed=42
dataset_name="coco" # coco | aokvqa | gqa
type="random" # random | popular | adversarial

# # llava
model="llava"
model_path="/data/share/wk/model_file/llava/llava-v1.5-7b"

# instructblip
# model="instructblip"
# model_path=/data/share/wk/model_file/instructblip

pope_path="/data/share/wk/POPE-main/output/${dataset_name}/${dataset_name}_pope_${type}.json"
# pope_path="/data/share/wk/POPE-main/output/seem/${dataset_name}/${dataset_name}_pope_seem_${type}.json"
data_path="/data/share/wk/dataset/val2014"
# data_path="/data/share/wk/dataset/images"
log_path="./logs"

use_glam=True
use_vcd=False
use_m3id=False

glam_alpha_pos=3
glam_alpha_neg=1
glam_beta=0.1

experiment_index=0
# name="ours"

#####################################
# Run single experiment
#####################################
export CUDA_VISIBLE_DEVICES=0
torchrun --nnodes=1 --nproc_per_node=1 --master_port 1234 eval_bench/pope_eval_${model}.py \
--seed ${seed} \
--model_path ${model_path} \
--model_base ${model} \
--pope_path ${pope_path} \
--data_path ${data_path} \
--log_path ${log_path} \
--use_glam ${use_glam} \
--use_vcd ${use_vcd} \
--use_m3id ${use_m3id} \
--glam_alpha_pos ${glam_alpha_pos} \
--glam_alpha_neg ${glam_alpha_neg} \
--glam_beta ${glam_beta} \
--experiment_index ${experiment_index} \
# --name ${name}