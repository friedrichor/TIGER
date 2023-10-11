import logging
from typing import Optional
from PIL import Image

import torch
import torch.nn as nn
from transformers import (T5Tokenizer, T5EncoderModel,
                          GPT2Tokenizer, GPT2Config, GPT2LMHeadModel)
from diffusers import StableDiffusionPipeline
from transformers.modeling_outputs import SequenceClassifierOutput


class ResponseModalPredictor(nn.Module):
    def __init__(self, pretrained_model_name_or_path, num_classes):
        super().__init__()
        self.backbone = T5EncoderModel.from_pretrained(pretrained_model_name_or_path)

        out_features = self.backbone.encoder.block[-1].layer[-1].DenseReluDense.wo.out_features
        self.fc1 = nn.Linear(out_features, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, input_ids, attention_mask):
        out_backbone = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        out_backbone = out_backbone.last_hidden_state  # [batch_size, seq_len, out_features]
        out_backbone = torch.mean(out_backbone, dim=1)  # [batch_size, out_features]
        out_fc1 = self.fc1(out_backbone)  # [batch_size, 128]
        out_fc2 = self.fc2(out_fc1)  # [batch_size, num_classes]

        return SequenceClassifierOutput(logits=out_fc2)
    
# TIGER Model (for Demonstration)
class TIGER(nn.Module):
    def __init__(
        self,
        rmp_pretrained_model_path_or_name: str,
        rmp_checkpoint: str,
        tdrg_pretrained_model_path_or_name: str,
        tdrg_checkpoint: str,
        t2it_model_path_or_name: str,
    ):
        super().__init__()
        
        logging.info("Loading Response Modal Predictor...")
        self.rmp_tokenizer, self.rmp_model = self._init_response_modal_predictor(rmp_pretrained_model_path_or_name, rmp_checkpoint)
        logging.info("Response Modal Predictor has been loaded.")
        
        logging.info("Loading Textual Dialogue Response Generator...")
        self.tdrg_tokenizer, self.tdrg_model = self._init_textual_dialogue_response_generator(tdrg_pretrained_model_path_or_name, tdrg_checkpoint)
        logging.info("Textual Dialogue Response Generator has been loaded.")
        
        logging.info("Loading Text-to-Image Translator...")
        self.t2it_model = self._init_text2image_translator(t2it_model_path_or_name)
        logging.info("Text-to-Image Translator has been loaded.")
    
    @property
    def device(self):
        return self.tdrg_model.device
        
    def _init_response_modal_predictor(self, rmp_pretrained_model_path_or_name, rmp_checkpoint):
        tokenizer = T5Tokenizer.from_pretrained(rmp_pretrained_model_path_or_name, truncation_side='left')
        tokenizer.add_special_tokens({'sep_token': '[SEP]'})

        model = ResponseModalPredictor(pretrained_model_name_or_path=rmp_pretrained_model_path_or_name, num_classes=2)
        model.load_state_dict(torch.load(rmp_checkpoint))
        logging.info("Load checkpoint from {}".format(rmp_checkpoint))
        
        return tokenizer, model
    
    def _init_textual_dialogue_response_generator(self, tdrg_pretrained_model_path_or_name, tdrg_checkpoint):
        tokenizer = GPT2Tokenizer.from_pretrained(tdrg_pretrained_model_path_or_name, truncation_side='left')
        tokenizer.add_tokens(['[UTT]', '[DST]'])

        config = GPT2Config.from_pretrained(tdrg_pretrained_model_path_or_name)
        if len(tokenizer) > config.vocab_size:
            config.vocab_size = len(tokenizer)

        model = GPT2LMHeadModel.from_pretrained(tdrg_checkpoint, config=config)
        
        return tokenizer, model
    
    def _init_text2image_translator(self, t2it_model_path_or_name):
        model = StableDiffusionPipeline.from_pretrained(t2it_model_path_or_name, torch_dtype=torch.float32)
        
        return model

    def response_modal_predict(self, context: str) -> bool:
        logging.info(f"context for response modal prediction: {context}")
        context_encoded = self.intent_predict_tokenizer.encode_plus(
            text=context,
            add_special_tokens=True,
            truncation=True,
            max_length=512,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = context_encoded['input_ids'].to(self.device)
        attention_mask = context_encoded['attention_mask'].to(self.device)
        
        pred_logits = self.rmp_model(
            input_ids=input_ids, 
            attention_mask=attention_mask
        ).logits
        pred_label = torch.max(pred_logits, dim=1)[1]
        logging.info(f"predicted label: {pred_label}")
    
        return True if pred_label else False
    
    def generate_textual_response(
        self, 
        context: str, 
        share_photo: bool,
        max_new_tokens: int = 512,
        min_new_tokens:int = 3,
        do_sample: bool = True,
        num_beams: int = 5,
        top_p: float = 0.9,
        top_k: float = 3,
        no_repeat_ngram_size: int = 3,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        temperature: float = 1.0,
    ) -> str:     
        tokenizer = self.tdrg_tokenizer
        tag_list = ["[UTT]", "[DST]"]  # Text responses begin with [UTT], image descriptions begin with [DST].
        tag_id_dic = {tag: tokenizer.convert_tokens_to_ids(tag) for tag in tag_list}
        tag = "[DST]" if share_photo else "[UTT]"
        bad_words = ["[UTT] [UTT]", "[UTT] [DST]", "[UTT] <|endoftext|>", "[DST] [UTT]", "[DST] [DST]", "[DST] <|endoftext|>"]
        
        input_ids = tokenizer.encode(
            context,
            add_special_tokens=False,
            return_tensors='pt'
        ).to(self.device)
        
        generated_ids = self.tdrg_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            top_p=top_p,
            top_k=top_k,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
            bad_words_ids=tokenizer(bad_words, add_prefix_space=True, add_special_tokens=False).input_ids,
            forced_decoder_ids=[[input_ids.shape[-1], tag_id_dic[tag]]],  # Specifies that the first token in the generated response is always the [tag]
            pad_token_id=tokenizer.eos_token_id, 
            eos_token_id=tokenizer.eos_token_id
        )
        
        generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

        end, i = 0, 0
        for i, token in enumerate(generated_tokens):
            if i == 0:  # Due to the definition of forced_decoder_ids, the first token of generated_tokens must be a [tag], so start with the second token.
                continue
            if token in tag_list:
                end = i
                break
        if end == 0 and i != 0:  # might not get a [tag]
            end = len(generated_tokens)
        
        response_tokens = generated_tokens[1:end]
        response_str = tokenizer.convert_tokens_to_string(response_tokens).lstrip()

        return response_str
    
    def generate_image(
        self,
        caption: str,
        negative_prompt: Optional[str] = None,
        seed: int = 42,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> Image.Image:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        image = self.t2it_model(
            prompt=caption,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        
        return image