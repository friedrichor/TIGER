import os
import sys
import gradio as gr
from datetime import datetime

import torch
from model import IntentPredictModel
from transformers import T5Tokenizer, GPT2LMHeadModel, GPT2Tokenizer
from diffusers import StableDiffusionPipeline


class Chat:
    def __init__(
        self, 
        intent_predict_model: IntentPredictModel, 
        intent_predict_tokenizer: T5Tokenizer,
        text_dialog_model: GPT2LMHeadModel,
        text_dialog_tokenizer: GPT2Tokenizer,
        text2image_model: StableDiffusionPipeline,
        device="cuda:0"
    ):
        self.intent_predict_model = intent_predict_model.to(device)
        self.intent_predict_tokenizer = intent_predict_tokenizer
        self.text_dialog_model = text_dialog_model.to(device)
        self.text_dialog_tokenizer = text_dialog_tokenizer
        self.text2image_model = text2image_model.to(device)
        self.device = device
        
        self.save_images_folder = os.path.join(sys.path[0], "generated_images")
        os.makedirs(self.save_images_folder, exist_ok=True)
        
        self.context_for_intent = ""
        self.context_for_text_dialog = ""
        
    def start_chat(self, chat_state):
        self.context_for_intent = ""
        self.context_for_text_dialog = ""
        
        return gr.update(interactive=True, placeholder='input the text (English).'), gr.update(value="Start Chat", interactive=False), gr.update(value="Restart Chat (Clear dialogue history)", interactive=True), chat_state
    
    def restart_chat(self, chat_state):
        self.context_for_intent = ""
        self.context_for_text_dialog = ""
        
        return None, gr.update(interactive=False, placeholder='Please click the <Start Chat> button to start chat!'), gr.update(value="Start Chat", interactive=True), gr.update(value="Restart Chat (Clear dialogue history)", interactive=False), chat_state
    
    def intent_predict(self, context: str):
        print(f"context for intent prediction: {context}")
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
        
        pred_logits = self.intent_predict_model(input_ids=input_ids, attention_mask=attention_mask).logits
        pred_label = torch.max(pred_logits, dim=1)[1]
    
        return True if pred_label else False
    
    def generate_response(self, context: str, share_photo: bool, decoding_stratey: str, num_beams , p_value, k_value):
        tokenizer = self.text_dialog_tokenizer
        tag_list = ["[UTT]", "[DST]"]  # 文本回复以 [UTT] 开头, 图像描述以 [DST] 开头
        tag_id_dic = {tag: tokenizer.convert_tokens_to_ids(tag) for tag in tag_list}
        tag = "[DST]" if share_photo else "[UTT]"
        bad_words = ["[UTT] [UTT]", "[UTT] [DST]", "[UTT] <|endoftext|>", "[DST] [UTT]", "[DST] [DST]", "[DST] <|endoftext|>"]
        
        input_ids = tokenizer.encode(
            context,
            add_special_tokens=False,
            return_tensors='pt'
        )
        
        if decoding_stratey == "Beam":
            generated_ids = self.text_dialog_model.generate(input_ids.to(self.device),
                                                            max_new_tokens=64, min_new_tokens=3,
                                                            do_sample=False, num_beams=num_beams, length_penalty=0.7,
                                                            no_repeat_ngram_size=3,
                                                            bad_words_ids=tokenizer(bad_words, add_prefix_space=True, add_special_tokens=False).input_ids,
                                                            forced_decoder_ids=[[input_ids.shape[-1], tag_id_dic[tag]]],  # 指定生成的回复中第一个token始终是tag(因为generated_ids中包括input_ids, 所以是第input_ids.shape[-1]位)  
                                                            pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
        elif decoding_stratey == "Top-P":
            generated_ids = self.text_dialog_model.generate(input_ids.to(self.device),
                                                            max_new_tokens=64, min_new_tokens=3,
                                                            do_sample=True, top_p=p_value,
                                                            no_repeat_ngram_size=3,
                                                            bad_words_ids=tokenizer(bad_words, add_prefix_space=True, add_special_tokens=False).input_ids,
                                                            forced_decoder_ids=[[input_ids.shape[-1], tag_id_dic[tag]]],  # 指定生成的回复中第一个token始终是tag(因为generated_ids中包括input_ids, 所以是第input_ids.shape[-1]位)  
                                                            pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
        elif decoding_stratey == "Top-K":
            generated_ids = self.text_dialog_model.generate(input_ids.to(self.device),
                                                            max_new_tokens=64, min_new_tokens=3,
                                                            do_sample=True, top_k=k_value,
                                                            no_repeat_ngram_size=3,
                                                            bad_words_ids=tokenizer(bad_words, add_prefix_space=True, add_special_tokens=False).input_ids,
                                                            forced_decoder_ids=[[input_ids.shape[-1], tag_id_dic[tag]]],  # 指定生成的回复中第一个token始终是tag(因为generated_ids中包括input_ids, 所以是第input_ids.shape[-1]位)  
                                                            pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
              
        generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

        end, i = 0, 0
        for i, token in enumerate(generated_tokens):
            if i == 0:  # 由于forced_decoder_ids的定义, generated_tokens第1个token必为tag, 故从第2个token开始
                continue
            if token in tag_list:
                end = i
                break
        if end == 0 and i != 0:  # 可能遇不到tag
            end = len(generated_tokens)
        
        response_tokens = generated_tokens[1:end]
        response_str = tokenizer.convert_tokens_to_string(response_tokens).lstrip()

        return response_str
    
    def respond(self, message, decoding_stratey, num_beams, p_value, k_value, text2image_seed, human_words, human_prompt, others_prompt, negative_prompt, chat_history, chat_state):
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        print("=" * 50)
        print(f"Time: {current_time}")
        print(f"User: {message}")
        # process context
        if self.context_for_intent == "":
            self.context_for_intent += message
        else:
            self.context_for_intent += " [SEP] " + message
        self.context_for_text_dialog += "[UTT] " + message
        
        share_photo = self.intent_predict(self.context_for_intent)
        response = self.generate_response(self.context_for_text_dialog, share_photo, decoding_stratey, num_beams, p_value, k_value)
        
        extra_prompt = {"human": human_prompt, "others": others_prompt}
        
        if share_photo:
            print(f"Generated Image Description: {response}")
            type = "others"
            for human_word in human_words:
                if human_word in response:
                    type = "human"
                    break
            caption = response + extra_prompt[type]
            
            generator = torch.Generator(device=self.device).manual_seed(int(text2image_seed))
            image = self.text2image_model(
                prompt=caption,
                negative_prompt=negative_prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
                generator=generator).images[0]
            
            save_image_path = f"{self.save_images_folder}/{response}.png"
            image.save(save_image_path)

            self.context_for_intent += " [SEP] " + response
            self.context_for_text_dialog += "[DST] " + response
            
            chat_history.append((message, (save_image_path, f"Generated Image Description: {response}")))

        else:
            print(f"Bot: {response}")
            self.context_for_intent += " [SEP] " + response
            self.context_for_text_dialog += "[UTT] " + response
            
            chat_history.append((message, response))
            
        return "", chat_history, chat_state
        