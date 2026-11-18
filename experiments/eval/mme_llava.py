import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math
import os
import tempfile

os.environ["TMPDIR"] = "/home/wk/wklab/shiyan/tmp"
tempfile.tempdir = "/home/wk/wklab/shiyan/tmp"

os.makedirs("/home/wk/wklab/shiyan/tmp", exist_ok=True)
# import kornia
from transformers import set_seed
from glam_utils.glam_sample import evolve_glam_sampling
from glam_utils.vcd_add_noise import add_diffusion_noise
evolve_glam_sampling()

from torchvision.transforms import v2
import random 

import warnings
warnings.filterwarnings(action='ignore')
from modify.modify_attention import llama_head_guide
flag=True

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for line in tqdm(questions):
    # for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        # conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        
        aug_dict = {
        'crop':v2.RandomResizedCrop(size=336),
        }
        aug_dict1 = {
        'rotation':v2.RandomRotation(degrees=180),
        }
        # For statistics
        pos_aug_counter = {k:0 for k in aug_dict}
        pos_aug_counter.update({None: 0})
    
        image_pos = None
        image_neg = None
        
        if args.use_glam:
            # ==============================================
            #              Image Transforms
            # ==============================================
            raw_image = image

            # positive augmentation
            pos_aug = random.choice(list(aug_dict.keys()))

            if pos_aug is not None:
                raw_image_pos = aug_dict[pos_aug](raw_image)

                image_pos = image_processor.preprocess(
                    raw_image_pos,
                    return_tensors='pt'
                )['pixel_values'][0]

                image_pos = torch.tensor(image_pos)

            # negative augmentation (different from pos_aug)
            # neg_aug_candidates = [k for k in aug_dict.keys() if k != pos_aug]
            neg_aug = random.choice(list(aug_dict1.keys()))

            if neg_aug is not None:
                raw_image_neg = aug_dict1[neg_aug](raw_image)

                image_neg = image_processor.preprocess(
                    raw_image_neg,
                    return_tensors='pt'
                )['pixel_values'][0]

                image_neg = torch.tensor(image_neg)

            pos_aug_counter[pos_aug] += 1


        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)




         
        #==============================================
        #             视觉注意力增强
        #==============================================
        num_image_tokens = (
            model.get_vision_tower().num_patches
        )
        image_pos1 = (
            input_ids[0] == IMAGE_TOKEN_INDEX
        ).nonzero(as_tuple=True)[0]

        img_start_idx = image_pos1.item()

        num_image_tokens = (
            model.get_vision_tower().num_patches
        )

        img_end_idx = (
            img_start_idx + num_image_tokens
        )

        print("img_start_idx:", img_start_idx)
        print("img_end_idx:", img_end_idx)



        if flag:
            llama_head_guide(
                model= model,
                guided_layer_range=[5, 18],
                aggregation="mean",
                alpha=0.5,
                img_start_idx=img_start_idx,
                img_end_idx=img_end_idx
            )



        with torch.inference_mode():
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    images_pos=(image_pos.unsqueeze(0).half().cuda() if image_pos is not None else None),
                    images_neg=(image_neg.unsqueeze(0).half().cuda() if image_neg is not None else None),
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_new_tokens=128,
                    use_cache=False,
                    use_glam=args.use_glam,
                    use_vcd=args.use_vcd,
                    use_m3id=args.use_m3id,
                    glam_alpha_pos=args.glam_alpha_pos,
                    glam_alpha_neg=args.glam_alpha_neg,
                    glam_beta=args.glam_beta,
                )
                
                
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        print("=" * 80)
        print("Question:", cur_prompt)
        print("Answer:", outputs)
        print("=" * 80)

        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "model_id": model_name,
                                   "image": image_file,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_glam", type=str2bool, default=False)

    parser.add_argument("--use_vcd", type=str2bool, default=False)
    parser.add_argument("--noise_step", type=int, default=500)
    
    parser.add_argument("--use_m3id", type=str2bool, default=False)
    
    parser.add_argument("--glam_alpha_pos", type=float, default=3)
    parser.add_argument("--glam_alpha_neg", type=float, default=1)
    parser.add_argument("--glam_beta", type=float, default=0.1)

    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
