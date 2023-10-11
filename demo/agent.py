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
        
        self.context_for_rmp = ""  # for response modal predictor
        self.context_for_tdrg = ""  # for textual dialogue response generator
        
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
        self.context_for_rmp = ""
        self.context_for_tdrg = ""
        
        return (
            gr.update(interactive=True, placeholder="input the text."),  # [input_text] Textbox
            gr.update(interactive=False),  # [start_btn] Button
            gr.update(interactive=True),  # [clear_btn] Button
            chat_state  # [chat_state] State
        )
    
    def restart_chat(self, chat_state):
        self.context_for_rmp = ""
        self.context_for_tdrg = ""
        
        return (
            None,  # [chatbot] Chatbot
            gr.update(interactive=False, placeholder="Please click the <Start Chat> button to start chat!"),  # [input_text] Textbox
            gr.update(interactive=True),  # [start] Button
            gr.update(interactive=False),  # [clear] Button
            chat_state  # [chat_state] State
        )
    
    def respond(
        self, 
        message,
        do_sample: bool,
        num_beams: int,
        top_p: float, 
        top_k: int, 
        text2image_seed: int,
        num_inference_steps: int,
        guidance_scale: float,
        human_words: str,
        human_prompt: str,
        others_prompt: str,
        negative_prompt: str,
        chat_history: gr.Chatbot, 
        chat_state: gr.State
    ):
        logging.info(f"User:\n{message}")
        # process context
        if self.context_for_rmp == "":
            self.context_for_rmp += message
        else:
            self.context_for_rmp += " [SEP] " + message
        self.context_for_tdrg += "[UTT] " + message
        
        share_photo = self.model.response_modal_predict(self.context_for_rmp)
        textual_response = self.model.generate_textual_response(
            self.context_for_tdrg, 
            share_photo,
            do_sample=do_sample,
            num_beams=num_beams,
            top_p=top_p,
            top_k=top_k
        )
        
        extra_prompt = {"human": human_prompt, "others": others_prompt}
        
        if share_photo:
            logging.info(f"Generated Image Description: {textual_response}")
            type = "others"
            for human_word in human_words:
                if human_word in textual_response:
                    type = "human"
                    break
            caption = textual_response + extra_prompt[type]
            # generate visual response
            visual_response = self.model.generate_image(
                caption, 
                negative_prompt, 
                seed=text2image_seed,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            # save 
            generated_image_save_path = os.path.join(self.generated_images_storage, "{}.jpg".format(len(os.listdir(self.generated_images_storage))))
            visual_response.save(generated_image_save_path)
            logging.info(f"generated image is saved to: {generated_image_save_path}")
            # update context
            self.context_for_rmp += " [SEP] " + textual_response
            self.context_for_tdrg += "[DST] " + textual_response
            
            chat_history.append((message, (generated_image_save_path, f"Generated Image Description: {textual_response}")))

        else:
            logging.info(f"\nBot : {textual_response}")
            # update context
            self.context_for_rmp += " [SEP] " + textual_response
            self.context_for_tdrg += "[UTT] " + textual_response
            
            chat_history.append((message, textual_response))
            
        return "", chat_history, chat_state
        