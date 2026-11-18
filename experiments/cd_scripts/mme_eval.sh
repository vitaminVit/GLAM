seed=5
dataset_name="mme"
question_file=/data/share/wk/dataset/MME_Benchmark_release_version/mme_hallucination.jsonl
image_folder="/data/share/wk/dataset/MME_Benchmark_release_version/MME_Benchmark"

# llava
# model="llava"
# model_path="/mnt/server8_hard1/donguk/NEURIPS2024/checkpoints/llava-v1.5-7b"

# instructblip
model="llava"
model_path="/data/share/wk/model_file/llava/llava-v1.5-7b"

gpu=0
export CUDA_VISIBLE_DEVICES=${gpu}

use_glam=True
use_vcd=False
use_m3id=False

glam_alpha_pos=3.0
glam_alpha_neg=1.0
glam_beta=0.1

log_path="./logs/mme/${model}"

python ./experiments/eval/mme_${model}.py \
--seed ${seed} \
--model-path ${model_path} \
--question-file ${question_file} \
--image-folder ${image_folder} \
--answers-file ./experiments/output/${model}_${dataset_name}_answers_seed${seed}.jsonl \
--use_glam ${use_glam} \
--use_vcd ${use_vcd} \
--use_m3id ${use_m3id} \
--glam_alpha_pos ${glam_alpha_pos} \
--glam_alpha_neg ${glam_alpha_neg} \
--glam_beta ${glam_beta} \

python ./experiments/eval/convert_answer_to_mme.py \
--output_path ./experiments/output/${model}_${dataset_name}_answers_seed${seed}.jsonl \
--seed ${seed} \
--model ${model} \
--log_path ${log_path}

python ./experiments/eval/eval_mme.py \
--results_dir ${log_path}/mme_answers \
--log_path ${log_path}
