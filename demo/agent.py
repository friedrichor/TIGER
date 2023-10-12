import os
import sys
import logging
from datetime import datetime
from omegaconf import OmegaConf

import gradio as gr

from tiger import TIGER


class ConversationalAgent:
    def __init__(self, config: OmegaConf) -> None:
        self.config = config
        
        self.model: TIGER = self.init_model()
        self.set_model_to_device()
        self.model.eval()
        
        self.generated_images_storage = os.path.join(self.outputs_dir, "generated")
        os.makedirs(self.generated_images_storage, exist_ok=True)
        
    @property
    def model_config(self):
        return self.config.model
    
    @property
    def run_config(self):
        return self.config.run
    
    @property
    def device(self):
        return self.run_config.device
    
    @property
    def outputs_dir(self):
        return self.run_config.outputs_dir
    
    def init_model(self):
        return TIGER(**self.model_config)
    
    def set_model_to_device(self):
        # move model to device
        self.model = self.model.to(self.device)
        self.model.t2it_model.to(self.device)
        
    def start_chat(self, chat_state):
        chat_state.append(["", "", ""])
        logging.info("=" * 30 + "Start Chat" + "=" * 30)
        
        return (
            None,  # [chatbot] Chatbot
            chat_state,  # [chat_state] State
            gr.update(interactive=True, placeholder="input the text."),  # [input_text] Textbox
            gr.update(interactive=False),  # [start_btn] Button
            gr.update(interactive=True),  # [clear_btn] Button
        )
    
    def restart_chat(self):
        logging.info("=" * 30 + "End Chat" + "=" * 30)
        
        return (
            None,  # [chatbot] Chatbot
            [],  # [chat_state] State
            gr.update(interactive=False, placeholder="Please click the <Start Chat> button to start chat!"),  # [input_text] Textbox
            gr.update(interactive=True),  # [start] Button
            gr.update(interactive=False),  # [clear] Button
        )
    
    def undo(self, chatbot, chat_state):
        logging.info("-" * 30 + "   Undo   " + "-" * 30)
        text_input, _ = chatbot.pop()
        logging.info(f"\n@ chatbot: {chatbot}")
        chat_state.pop()
        logging.info(f"\n@ chat_state: {chat_state}")
        logging.info("-" * 70)
        
        return text_input, chatbot, chat_state
    
    def generate_visual_response(
        self,
        description: str,
        num_inference_steps: int,
        guidance_scale: float,
        human_words,
        human_prompt: str,
        others_prompt: str,
        negative_prompt: str
    ) -> str:
        logging.info(f"\n@@ Generated Image Description: {description}")
        extra_prompt = {"human": human_prompt, "others": others_prompt}
        type = "others"
        for human_word in human_words:
            if human_word in description:
                type = "human"
                break
        caption = description + extra_prompt[type]
        logging.info(f"\n@@ Caption for Generation = {caption}")
        
        visual_response = self.model.generate_image(
            caption, 
            negative_prompt, 
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        
        generated_image_save_path = os.path.join(self.generated_images_storage, "{}.jpg".format(len(os.listdir(self.generated_images_storage))))
        visual_response.save(generated_image_save_path)
        logging.info(f"@@ Generated Image is saved to: {generated_image_save_path}")
        
        return generated_image_save_path
    
    def respond(
        self, 
        message,
        chat_history: gr.Chatbot,
        chat_state: gr.State,
        do_sample: bool,
        num_beams: int,
        top_p: float, 
        top_k: int, 
        num_inference_steps: int,
        guidance_scale: float,
        human_words: str,
        human_prompt: str,
        others_prompt: str,
        negative_prompt: str,
    ):
        logging.info(f"\n@@ User Input: {message}")
        # process context
        # print(f"$$ chat_state: {chat_state}")
        _, context_for_rmp, context_for_tdrg = chat_state[-1]
        if context_for_rmp == "":
            context_for_rmp += message
        else:
            context_for_rmp += " [SEP] " + message
        context_for_tdrg += "[UTT] " + message
        
        logging.info(f"\n@@ context_for_rmp = {context_for_rmp}")
        logging.info(f"\n@@ context_for_tdrg = {context_for_tdrg}")
        
        logging.info(f"## Textual Response Generation Setting")
        logging.info(f"do_sample = {do_sample}")
        logging.info(f"num_beams = {num_beams}")
        logging.info(f"top_p = {top_p}")
        logging.info(f"top_k = {top_k}")
        logging.info(f"## Visual Response Generation Setting")
        logging.info(f"num_inference_steps = {num_inference_steps}")
        logging.info(f"guidance_scale = {guidance_scale}")
        
        share_photo = True if self.model.response_modal_predict(context_for_rmp) else False
        textual_response = self.model.generate_textual_response(
            context_for_tdrg, 
            share_photo,
            do_sample=do_sample,
            num_beams=num_beams,
            top_p=top_p,
            top_k=top_k
        )
        
        if share_photo:
            generated_image_path = self.generate_visual_response(
                description=textual_response,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                human_words=human_words,
                human_prompt=human_prompt,
                others_prompt=others_prompt,
                negative_prompt=negative_prompt
            )
            # update context
            context_for_rmp += " [SEP] " + textual_response
            context_for_tdrg += "[DST] " + textual_response
            
            chat_history.append((message, (f'''<img src="./file={generated_image_path}" style="display: inline-block;">''')))

        else:
            # update context
            context_for_rmp += " [SEP] " + textual_response
            context_for_tdrg += "[UTT] " + textual_response
            
            logging.info(f"***** Second Prediction *****")
            logging.info(f"\n@@ context_for_rmp = {context_for_rmp}")
            logging.info(f"\n@@ context_for_tdrg = {context_for_tdrg}")
            share_photo = True if self.model.response_modal_predict(context_for_rmp) else False
            if share_photo:
                description = self.model.generate_textual_response(
                    context_for_tdrg, 
                    share_photo,
                    do_sample=do_sample,
                    num_beams=num_beams,
                    top_p=top_p,
                    top_k=top_k
                )
                generated_image_path = self.generate_visual_response(
                    description=description,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    human_words=human_words,
                    human_prompt=human_prompt,
                    others_prompt=others_prompt,
                    negative_prompt=negative_prompt
                )
                # update context
                context_for_rmp += " [SEP] " + description
                context_for_tdrg += "[DST] " + description
                
                chat_history.append((message, (f'''{textual_response}\n<img src="./file={generated_image_path}" style="display: inline-block;">''')))
            else:
                chat_history.append((message, textual_response))
        
        chat_state.append([message, context_for_rmp, context_for_tdrg])
        
        logging.info(f"\n@@ chat_history: {chat_history}")
        # logging.info(f"\n@@ chat_state: {chat_state}")
            
        return "", chat_history, chat_state, gr.update(interactive=True)
        