import json
import re
from typing import Dict, List, Tuple, Optional
from PIL import Image
import torch


class LocalAugmentationSelector:
    def __init__(
        self,
        tokenizer,
        model,
        image_processor,
        conv_mode: str = "llava_v1",
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ):
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.constants import DEFAULT_IMAGE_TOKEN
        
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.conv_mode = conv_mode
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        
        self.available_augmentations = [
            "horizontal flip",
            "vertical flip", 
            "rotation",
            "color jitter",
            "gaussian blur",
            "crop"
        ]
        
        self._conv_template = conv_templates[conv_mode].copy()
    
    def _build_prompt(self, query: str) -> str:
        aug_list = "\n".join([
            f"- {name}" for name in self.available_augmentations
        ])
        
        prompt = f"""Available image transformation methods:
{aug_list}

Current question to answer: {query}

Based on the question above, select two different image transformations to improve model accuracy:
1. image_pos: A transformation that helps the model better understand image content and answer correctly
2. image_neg: A different transformation from image_pos

Return in JSON format with only the transformation names:
{{
    "image_pos": "transformation_name",
    "image_neg": "transformation_name"
}}

Return only JSON, no other content."""
        
        return prompt
    
    def _parse_response(self, response: str) -> Tuple[str, str]:
        try:
            match = re.search(r'\{[^{}]*"image_pos"\s*:\s*"([^"]+)"\s*,\s*"image_neg"\s*:\s*"([^"]+)"[^}]*\}', response, re.DOTALL)
            if not match:
                match = re.search(r'\{[^{}]*"pos"\s*:\s*"([^"]+)"\s*,\s*"neg"\s*:\s*"([^"]+)"[^}]*\}', response, re.DOTALL)
            
            if match:
                pos_aug = match.group(1).strip()
                neg_aug = match.group(2).strip()
                
                if pos_aug not in self.available_augmentations:
                    pos_aug = "horizontal flip"
                if neg_aug not in self.available_augmentations:
                    neg_aug = "vertical flip"
                
                if pos_aug == neg_aug:
                    available_neg = [a for a in self.available_augmentations if a != pos_aug]
                    neg_aug = available_neg[0] if available_neg else "vertical flip"
                
                return pos_aug, neg_aug
        except Exception as e:
            print(f"Failed to parse response: {e}")
        
        return "horizontal flip", "vertical flip"
    
    def _generate_with_image(self, prompt: str, image: Image.Image) -> str:
        from llava.conversation import SeparatorStyle
        from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX
        
        try:
            image_tensor = process_images([image], self.image_processor, self.model.config)
            if isinstance(image_tensor, (list, tuple)) and len(image_tensor) > 0:
                image_tensor = image_tensor[0]
            image_tensor = image_tensor.half().cuda()
        except Exception as e:
            print(f"Image processing failed: {e}")
            raise
        
        self._conv_template.messages = []
        full_prompt = self.DEFAULT_IMAGE_TOKEN + "\n" + prompt
        self._conv_template.append_message(self._conv_template.roles[0], full_prompt)
        self._conv_template.append_message(self._conv_template.roles[1], None)
        full_prompt = self._conv_template.get_prompt()
        
        input_ids = tokenizer_image_token(full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        stop_str = self._conv_template.sep if self._conv_template.sep_style != SeparatorStyle.TWO else self._conv_template.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        
        try:
            with torch.inference_mode():
                with torch.no_grad():
                    output_ids = self.model.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0),
                        do_sample=True,
                        temperature=self.temperature,
                        max_new_tokens=self.max_new_tokens,
                        stopping_criteria=[stopping_criteria],
                    )
        except Exception as e:
            print(f"Model generation failed: {e}")
            raise
        
        try:
            input_token_len = input_ids.shape[1]
            outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
            if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                outputs = outputs[0]
            else:
                outputs = ""
            outputs = outputs.strip()
            
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
        except Exception as e:
            print(f"Decoding failed: {e}")
            raise
        
        return outputs
    
    def _tokenize_with_image_token(self, prompt: str):
        from llava.constants import IMAGE_TOKEN_INDEX
        
        prompt_chunks = [self.tokenizer(chunk).input_ids for chunk in prompt.split(self.DEFAULT_IMAGE_TOKEN)]
        
        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]
        
        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == self.tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])
        
        for x in insert_separator(prompt_chunks, [IMAGE_TOKEN_INDEX] * (offset + 1)):
            input_ids.extend(x[offset:])
        
        return torch.tensor(input_ids, dtype=torch.long)
    
    def select_augmentations(self, query: str, image: Image.Image) -> Tuple[str, str]:
        try:
            prompt = self._build_prompt(query)
            response = self._generate_with_image(prompt, image)
            
            if not response or len(response) == 0:
                print("Empty response from model, using default augmentations")
                return "horizontal flip", "vertical flip"
            
            return self._parse_response(response)
        except Exception as e:
            print(f"LLM generation failed: {e}, using default augmentations")
            import traceback
            traceback.print_exc()
            return "horizontal flip", "vertical flip"


def create_local_augmentation_selector(
    tokenizer,
    model,
    image_processor,
    conv_mode: str = "llava_v1",
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> Optional[LocalAugmentationSelector]:
    try:
        return LocalAugmentationSelector(
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
            conv_mode=conv_mode,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    except Exception as e:
        print(f"Failed to create selector: {e}")
        return None
