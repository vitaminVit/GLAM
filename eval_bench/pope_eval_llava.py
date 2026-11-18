import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
#===================================================
import os
import tempfile

os.environ["TMPDIR"] = "/home/wk/wklab/shiyan/tmp"
tempfile.tempdir = "/home/wk/wklab/shiyan/tmp"

os.makedirs("/home/wk/wklab/shiyan/tmp", exist_ok=True)
#===========================================================

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/experiments')
# print(sys.path)

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import Conversation, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

from utils import dist_util
from utils.logger import create_logger
from glob import glob

import re
from PIL import Image
from torchvision.transforms import v2

from pope_loader import POPEDataSet

# import kornia
from glam_utils.glam_sample import evolve_glam_sampling
from glam_utils.vcd_add_noise import add_diffusion_noise

#=========================我的包
from modify.modify_attention import llama_head_guide


evolve_glam_sampling()
torch.multiprocessing.set_sharing_strategy('file_system')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description="POPE evaluation on LVLMs.")
    parser.add_argument("--model_path", type=str, default="/mnt/server8_hard1/donguk/checkpoints/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default=None)
    
    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    
    parser.add_argument("--data_path", type=str, default="/mnt/server18_hard0/jhjang/LVLM/crg/data/coco/val2014")
    parser.add_argument("--pope_path", type=str, default="/mnt/server8_hard1/donguk/rips2024/experiments/data/POPE/coco/coco_pope_random.json")
    parser.add_argument("--log_path", type=str, default="/mnt/server16_hard0/sangmin/code/neurips2024/logs/pope")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--use_glam", type=str2bool, default=False)

    parser.add_argument("--use_vcd", type=str2bool, default=False)
    parser.add_argument("--noise_step", type=int, default=500)
    
    parser.add_argument("--use_m3id", type=str2bool, default=False)
    
    parser.add_argument("--glam_alpha_pos", type=float, default=3)
    parser.add_argument("--glam_alpha_neg", type=float, default=1)
    parser.add_argument("--glam_beta", type=float, default=0.1)
    
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--experiment_index", type=int, default=0)
    parser.add_argument("--name", type=str, default="pope_eval_llava")

    args = parser.parse_args()
    return args


def print_acc(pred_list, label_list, logger):
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)
    # unknown_ratio = pred_list.count(2) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    logger.info('TP\tFP\tTN\tFN\t')
    logger.info('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)

    return acc, precision, recall, f1, yes_ratio


def recorder(out, pred_list):
    NEG_WORDS = ["No", "not", "no", "NO"]
    for line in out.split('\n'):

        line = line.replace('.', '')
        line = line.replace(',', '')
        words = line.split(' ')

        if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
            pred_list.append(0)
        else:
            pred_list.append(1)
    
    return pred_list


def main():
    args = parse_args()
    # Setup DDP:
    dist_util.setup_dist(args)
    device = dist_util.device()
    
    # flag = False
    flag = True

    # Setup an experiment folder:
    if dist.get_rank() == 0:
        os.makedirs(
            args.log_path, exist_ok=True
        )  # Make results folder (holds all experiment subfolders)
        model_string_name = args.model_path.split("/")[-1]
        # experiment_index = len(glob(f"{args.log_path}/{model_string_name}/*")) + args.experiment_index
        # experiment_index = args.experiment_index
        # experiment_dir = f"{args.log_path}/{model_string_name}/{experiment_index:03d}"  # Create an experiment folder
        experiment_dir = f"{args.log_path}/{model_string_name}/{args.glam_alpha_pos}_{args.glam_alpha_neg}_{args.glam_beta}"  # Create an experiment folder
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # ========================================
    #             Model & Dataset
    # ========================================
    logger.info('Initializing Model')

    #### for glam
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

    pope_dataset = POPEDataSet(
        pope_path=args.pope_path, 
        data_path=args.data_path,
        trans=image_processor,
        model=args.model_base
    )
    pope_loader = torch.utils.data.DataLoader(
        pope_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        drop_last=False
    )

    # ==============================================
    #               Augmentations
    # ==============================================

    aug_dict = {
    'crop':v2.RandomResizedCrop(size=336),
    }
    # aug_dict1 = {
    # 'rotation':v2.RandomRotation(degrees=180),
    # }
    
    
    # For statistics
    pos_aug_counter = {k:0 for k in aug_dict}
    pos_aug_counter.update({None: 0})

    # ========================================
    #            Start Generation
    # ========================================
    logger.info("Start eval...")
    pred_list, label_list = [], []
    for batch_id, data in tqdm(enumerate(pope_loader), total=len(pope_loader)):
        image = data["image"][0]
        qs = data["query"][0]
        label = data["label"]
        image_path = data["image_path"][0]
        label_list = label_list + list(label)

        image_pos = None
        image_neg = None

        # if args.use_glam:
        #     # ==============================================
        #     #              Image Transforms
        #     # ==============================================
        #     raw_image = Image.open(image_path)
        #     pos_aug = random.choice(list(aug_dict.keys()))

        #     if pos_aug is not None:
        #         raw_image_pos = aug_dict[pos_aug](raw_image)
        #         image_pos = image_processor.preprocess(raw_image_pos, return_tensor='pt')['pixel_values'][0] 
        #         image_pos = torch.tensor(image_pos)
        
        #     pos_aug_counter[pos_aug] += 1
        #     logger.info(f"GLAM Transformation: {pos_aug}")
        if args.use_glam:
            # ==============================================
            #              Image Transforms
            # ==============================================
            raw_image = Image.open(image_path)

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
            # neg_aug = random.choice(list(aug_dict1.keys()))

            # if neg_aug is not None:
            #     raw_image_neg = aug_dict1[neg_aug](raw_image)

            #     image_neg = image_processor.preprocess(
            #         raw_image_neg,
            #         return_tensors='pt'
            #     )['pixel_values'][0]

            #     image_neg = torch.tensor(image_neg)

            # pos_aug_counter[pos_aug] += 1

            logger.info(f"GLAM Positive Transformation: {pos_aug}")
            # logger.info(f"GLAM Negative Transformation: {neg_aug}")

        elif args.use_vcd:
            image_neg = add_diffusion_noise(image, args.noise_step)
            
        
        
        # ==============================================
        #              Text prompt setting
        # ==============================================
        conv_out = Conversation(
            system="A chat between a curious human and an artificial intelligence assistant. "
                   "The assistant gives helpful, detailed, and polite answers to the human's questions.",
            roles=("USER", "ASSISTANT"),
            version="v1",
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.TWO,
            sep=" ",
            sep2="</s>",
        )
        
        qu_out = DEFAULT_IMAGE_TOKEN + '\n' + qs # for opera? setting
        # qu_out = DEFAULT_IMAGE_TOKEN + '\n' + qs + " Please answer this question with one word." # for VCD setting
        conv_out.append_message(conv_out.roles[0], qu_out)
        conv_out.append_message(conv_out.roles[1], None)
        prompt_out = conv_out.get_prompt()
        
        input_ids = tokenizer_image_token(prompt_out, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv_out.sep if conv_out.sep_style != SeparatorStyle.TWO else conv_out.sep2

        
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



        # ==============================================
        #                glam method
        # ==============================================
        with torch.inference_mode():
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    images=image.unsqueeze(0).half().cuda(),
                    images_pos=(image_pos.unsqueeze(0).half().cuda() if image_pos is not None else None),
                    images_neg=(image_neg.unsqueeze(0).half().cuda() if image_neg is not None else None),
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=False,  # 必须关闭缓存才能动态修改attention权重
                    use_glam=args.use_glam,
                    use_vcd=args.use_vcd,
                    use_m3id=args.use_m3id,
                    glam_alpha_pos=args.glam_alpha_pos,
                    glam_alpha_neg=args.glam_alpha_neg,
                    glam_beta=args.glam_beta
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
        pred_list = recorder(outputs, pred_list)

        # =========================================================
        #                     日志记录
        # =========================================================
        logger.info(f"[VQA for glam]")
        logger.info(f"V: {image_path}")
        logger.info(f"Q: {qs}")
        logger.info(f"A: {outputs}")
        if label == 1:
            logger.info(f"GT: Yes")
        elif label == 0:
            logger.info(f"GT: No")
        logger.info(f"=" * 50)


        # =========================================================
        #                  实时统计打印
        # =========================================================
        if dist.get_rank() == 0:

            # 当前预测
            current_pred = pred_list[-1]
            current_label = label.item() if torch.is_tensor(label) else label

            # 当前样本是否正确
            if current_pred == 1 and current_label == 1:
                current_result = "TP"
            elif current_pred == 1 and current_label == 0:
                current_result = "FP"
            elif current_pred == 0 and current_label == 0:
                current_result = "TN"
            else:
                current_result = "FN"

            # 实时统计
            TP, TN, FP, FN = 0, 0, 0, 0

            for pred, gt in zip(pred_list, label_list[:len(pred_list)]):

                if pred == 1 and gt == 1:
                    TP += 1

                elif pred == 1 and gt == 0:
                    FP += 1

                elif pred == 0 and gt == 0:
                    TN += 1

                elif pred == 0 and gt == 1:
                    FN += 1

            # 实时指标
            total = TP + TN + FP + FN

            acc = (TP + TN) / total if total > 0 else 0

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0

            recall = TP / (TP + FN) if (TP + FN) > 0 else 0

            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0
            )

            yes_ratio = pred_list.count(1) / len(pred_list)

            # tqdm安全打印（不会打乱进度条）
            print("\n" + "=" * 70)

            print(f"[Sample {batch_id}]")

            print(f"Image      : {os.path.basename(image_path)}")

            print(f"Question   : {qs}")

            print(f"Prediction : {outputs}")

            print(f"GT         : {'Yes' if current_label == 1 else 'No'}")

            print(f"Result     : {current_result}")

            print("-" * 70)

            print(
                f"TP={TP} | FP={FP} | TN={TN} | FN={FN}"
            )

            print(
                f"ACC={acc*100:.2f}% | "
                f"Precision={precision*100:.2f}% | "
                f"Recall={recall*100:.2f}% | "
                f"F1={f1*100:.2f}% | "
                f"YesRatio={yes_ratio*100:.2f}%"
            )

            print("=" * 70)
            # =========================================================
            #                 保存到 jsonl
            # =========================================================

            save_dict = {
            
                # 基本信息
                "sample_id": batch_id,
                "image_path": image_path,
                # "question": qs,

                # # 模型输出
                # "prediction_text": outputs,

                # # label
                # "gt_label": int(current_label),
                # "pred_label": int(current_pred),

                # # 当前结果
                # "result": current_result,

                # # confusion matrix
                # "TP": TP,
                # "FP": FP,
                # "TN": TN,
                # "FN": FN,

                # 实时指标
                "ACC": acc,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "YesRatio": yes_ratio,

                # # glam
                # "use_glam": args.use_glam,
                # "use_vcd": args.use_vcd,

                # # augment
                # "positive_aug": pos_aug if args.use_glam else None,
            }

            with open(f"/home/wk/wklab/shiyan/GLAM_wk/chair_results/llava/{args.name}.jsonl", "a", encoding="utf-8") as f:
            
                f.write(
                    json.dumps(save_dict, ensure_ascii=False) + "\n"
                )
            
    if len(pred_list) != 0:
        logger.info(vars(args))
        # logger.info("Prompt for Aug:", prompt_aug)
        # logger.info("Prompt for glam:", prompt_out)
        acc, precision, recall, f1, yes_ratio = print_acc(pred_list, label_list, logger)
        
        acc = round(acc*100,2)
        precision = round(precision*100,2)
        recall = round(recall*100,2)
        f1 = round(f1*100,2)
        yes_ratio = round(yes_ratio*100,2)
        
        logger.info(
            f"acc: {acc}, precision: {precision}, recall: {recall}, f1: {f1}, yes_ratio: {yes_ratio}"
        )
        if args.use_glam:
            logger.info(f"GLAM Transformation: {pos_aug_counter}")

if __name__ == "__main__":
    main()
