import argparse
import gradio as gr

from demo import CustomTheme, ConversationalAgent
from utils import load_yaml, init_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str, 
        default="demo/demo_config.yaml"
    )
    args = parser.parse_args()
    
    return args


def main(args):
    config = load_yaml(args.config)
    # logging
    init_logger(config)
    
    agent = ConversationalAgent(config)
    theme = CustomTheme()
    
    title_demo = """<center><B><font face="Kalam:wght@700" size=5>A Multimodal Dialogue System based on TIGER</font></B></center>"""
    title_paper = """<center><B><font face="Courier" size=4>TIGER: A Unified Generative Model Framework for Multimodal Dialogue Response Generation</font></B></center>"""
    description = """Input text start chatting!"""
    language = """Language: English"""
    
    with gr.Blocks(theme) as demo_chatbot:
        gr.Markdown(title_demo)
        gr.Markdown(title_paper)
        gr.Markdown(description)
        gr.Markdown(language)
        
        gr.Markdown("Prompt templates")
        with gr.Row():
            human_words = gr.Textbox(value='''"man", "men", "woman", "women", "people", "person", "human", "male", "female", "boy", "girl", "child", "kid", "baby", "player"''',
                                     interactive=True,
                                     label="Human Words",
                                     show_copy_button=True)
            human_prompt = gr.Textbox(value=", facing the camera, photograph, highly detailed face, depth of field, moody light, style by Yasmin Albatoul, Harry Fayt, centered, extremely detailed, Nikon D850, award winning photography",
                                      interactive=True,
                                      label="Human Extra Prompts",
                                      show_copy_button=True)
        with gr.Row():
            gr.Textbox(placeholder="No need to fill",
                       interactive=False,
                       label="Others Words")
            others_prompt = gr.Textbox(value=", depth of field. bokeh. soft light. by Yasmin Albatoul, Harry Fayt. centered. extremely detailed. Nikon D850, (35mm|50mm|85mm). award winning photography.",
                                       interactive=True,
                                       label="Others Extra Prompts",
                                       show_copy_button=True)
            
        negative_prompt = gr.Textbox(value="cartoon, anime, ugly, (bad proportions, unnatural feature, incongruous feature:1.4), (blurry, un-sharp, fuzzy, un-detailed skin:1.2), (facial contortion, poorly drawn face, deformed iris, deformed pupils:1.3), (mutated hands and fingers:1.5), disconnected hands, disconnected limbs",
                                     interactive=True,
                                     label="Negative Prompt",
                                     show_copy_button=True)

        with gr.Row():
            with gr.Column(scale=3):
                start_btn = gr.Button("Start Chat", variant="primary", interactive=True)
                clear_btn = gr.Button("Clear Context", interactive=False)
                
                with gr.Accordion("Text Generation Settings"):
                    do_sample = gr.Radio(["True", "False"],
                                         value="True",
                                         interactive=True,
                                         label="whether to do sample")
                    num_beams = gr.Slider(minimum=1, maximum=10, step=1,
                                          value=5,
                                          interactive=True,
                                          label="beam search size")
                    top_p = gr.Slider(minimum=0, maximum=1, step=0.01,
                                      value=0.9,
                                      interactive=True,
                                      label='"p" value of Top-P')
                    top_k = gr.Slider(minimum=1, maximum=10, step=1,
                                      value=3,
                                      interactive=True,
                                      label='"k" value of Top-K')
                    
                with gr.Accordion("Image Generation Settings"):
                    text2image_seed = gr.Number(value=42,
                                                interactive=True,
                                                label="seed for text2image")
                    num_inference_steps = gr.Slider(minimum=20, maximum=70, step=1,
                                                    value=20,
                                                    interactive=True,
                                                    label='inference steps',
                                                    visible=False)
                    guidance_scale = gr.Slider(minimum=5, maximum=10, step=0.1,
                                               value=7.5,
                                               interactive=True,
                                               label='guidance scale',
                                               visible=False)
                    
                

            with gr.Column(scale=7):
                chat_state = gr.State()
                chatbot = gr.Chatbot(label="TIGER")
                text_input = gr.Textbox(label="User", placeholder="Please click the <Start Chat> button to start chat!", interactive=False)
        
        start_btn.click(agent.start_chat, [chat_state], [text_input, start_btn, clear_btn, chat_state])
        clear_btn.click(agent.restart_chat, [chat_state], [chatbot, text_input, start_btn, clear_btn, chat_state], queue=False)
        text_input.submit(
            fn=agent.respond, 
            inputs=[text_input, do_sample, num_beams, top_p, top_k, text2image_seed, num_inference_steps, guidance_scale, human_words, human_prompt, others_prompt, negative_prompt, chatbot, chat_state], 
            outputs=[text_input, chatbot, chat_state]
        )
        
    demo_chatbot.launch(share=True, enable_queue=False, server_name="127.0.0.1")


if __name__ == "__main__":
    args = parse_args()

    main(args)
