import os
import sys
import argparse
import gradio as gr

import torch
from model import IntentPredictModel
from transformers import (T5Tokenizer, 
                          GPT2Tokenizer, GPT2Config, GPT2LMHeadModel)
from diffusers import StableDiffusionPipeline

from chatbot import Chat
from component import decoding_setting


def main(args):
    # Intent Prediction
    print("Loading Intent Prediction Classifier...")
    ## tokenizer
    intent_predict_tokenizer = T5Tokenizer.from_pretrained(args.intent_predict_model_name, truncation_side='left')
    intent_predict_tokenizer.add_special_tokens({'sep_token': '[SEP]'})
    # model
    intent_predict_model = IntentPredictModel(pretrained_model_name_or_path=args.intent_predict_model_name, num_classes=2)
    intent_predict_model.load_state_dict(torch.load(args.intent_predict_model_weights_path, map_location=args.device))
    print("Intent Prediction Classifier loading completed.")
    
    # Textual Dialogue Response Generator
    print("Loading Textual Dialogue Response Generator...")
    ## tokenizer
    text_dialog_tokenizer = GPT2Tokenizer.from_pretrained(args.text_dialog_model_name, truncation_side='left')
    text_dialog_tokenizer.add_tokens(['[UTT]', '[DST]'])
    print(len(text_dialog_tokenizer))
    # config
    text_dialog_config = GPT2Config.from_pretrained(args.text_dialog_model_name)
    if len(text_dialog_tokenizer) > text_dialog_config.vocab_size:
        text_dialog_config.vocab_size = len(text_dialog_tokenizer)
    # load model weights
    text_dialog_model = GPT2LMHeadModel.from_pretrained(args.text_dialog_model_weights_path, config=text_dialog_config)
    print("Textual Dialogue Response Generator loading completed.")

    # Text-to-Image Translator
    print("Loading Text-to-Image Translator...")
    text2image_model = StableDiffusionPipeline.from_pretrained(args.text2image_model_weights_path, torch_dtype=torch.bfloat16)
    print("Text-to-Image Translator loading completed.")
    
    chat = Chat(intent_predict_model, intent_predict_tokenizer, 
                text_dialog_model, text_dialog_tokenizer,
                text2image_model, 
                args.device)
    
    title = """<h1 align="center">A Multimodal Dialogue System based on Tiger</h1>"""
    description1 = """<h2>Input text start chatting!</h2>"""
    hr = """<hr>"""
    language = """<h3>Language: English</h3>"""

    with gr.Blocks() as demo_chatbot:
        gr.Markdown(title)
        gr.Markdown(description1)
        gr.Markdown(hr)
        gr.Markdown(language)
        gr.Markdown(hr)
        
        gr.Markdown("Prompt templates")
        with gr.Row():
            human_words = gr.Textbox(value='''"man", "men", "woman", "women", "people", "person", "human", "male", "female", "boy", "girl", "child", "kid", "baby", "player"''',
                                     interactive=True,
                                     label="Human Words")
            human_prompt = gr.Textbox(value=", facing the camera, photograph, highly detailed face, depth of field, moody light, style by Yasmin Albatoul, Harry Fayt, centered, extremely detailed, Nikon D850, award winning photography",
                                      interactive=True,
                                      label="Human Extra Prompts")
        with gr.Row():
            gr.Textbox(placeholder="No need to fill",
                       interactive=False,
                       label="Others Words")        
            others_prompt = gr.Textbox(value=", depth of field. bokeh. soft light. by Yasmin Albatoul, Harry Fayt. centered. extremely detailed. Nikon D850, (35mm|50mm|85mm). award winning photography.",
                                       interactive=True,
                                       label="Others Extra Prompts")
            
        negative_prompt = gr.Textbox(value="cartoon, anime, ugly, (bad proportions, unnatural feature, incongruous feature:1.4), (blurry, un-sharp, fuzzy, un-detailed skin:1.2), (facial contortion, poorly drawn face, deformed iris, deformed pupils:1.3), (mutated hands and fingers:1.5), disconnected hands, disconnected limbs",
                                     interactive=True,
                                     label="Negative Prompt")                   

        with gr.Row():
            with gr.Column(scale=0.4):
                decoding_stratey = gr.Radio(["Beam", "Top-P", "Top-K"],
                                            interactive=True,
                                            label="Decoding Strategy")
                num_beams = gr.Slider(minimum=1, maximum=10, step=1,
                                      value=5,
                                      interactive=True,
                                      label="beam search size",
                                      visible=False)
                p_value = gr.Slider(minimum=0, maximum=1, step=0.01,
                                    value=0.8,
                                    interactive=True,
                                    label='"p" value of Top-P',
                                    visible=False)
                k_value = gr.Slider(minimum=1, maximum=20, step=1,
                                    value=3,
                                    interactive=True,
                                    label='"k" value of Top-K',
                                    visible=False)
                
                text2image_seed = gr.Number(value=41,
                                            interactive=True, 
                                            label="seed for text2image")
                    
                
                start = gr.Button("Start Chat", variant="primary")
                clear = gr.Button("Restart Chat (Clear Dialogue History)", interactive=False)

            with gr.Column():
                chat_state = gr.State()
                chatbot = gr.Chatbot(label='Tiger')
                text_input = gr.Textbox(label='User', placeholder="Please click the <Start Chat> button to start chat!", interactive=False)
        
        
        decoding_stratey.change(fn=decoding_setting, inputs=[decoding_stratey, num_beams, p_value, k_value], outputs=[num_beams, p_value, k_value])
        start.click(chat.start_chat, [chat_state], [text_input, start, clear, chat_state])
        text_input.submit(chat.respond, inputs=[text_input, decoding_stratey, num_beams, p_value, k_value, text2image_seed, human_words, human_prompt, others_prompt, negative_prompt, chatbot, chat_state], outputs=[text_input, chatbot, chat_state])
        clear.click(chat.restart_chat, [chat_state], [chatbot, text_input, start, clear, chat_state], queue=False)

    demo_chatbot.launch(share=True, enable_queue=False, server_name="127.0.0.1")


if __name__ == "__main__":
    intent_predict_model_weights_path = os.path.join(sys.path[0], "model_weights/Tiger_t5_base_encoder.pth")
    text_dialog_model_weights_path = os.path.join(sys.path[0], "model_weights/Tiger_DialoGPT_medium.pth")
    text2image_model_weights_path = "friedrichor/stable-diffusion-2-1-realistic"
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--intent_predict_model_name', type=str, default="t5-base")
    parser.add_argument('--intent_predict_model_weights_path', type=str, default=intent_predict_model_weights_path)

    parser.add_argument('--text_dialog_model_name', type=str, default="microsoft/DialoGPT-medium")
    parser.add_argument('--text_dialog_model_weights_path', type=str, default=text_dialog_model_weights_path)
    
    parser.add_argument('--text2image_model_weights_path', type=str, default=text2image_model_weights_path)

    parser.add_argument('--device', default="cuda:0")

    args = parser.parse_args()

    main(args)
