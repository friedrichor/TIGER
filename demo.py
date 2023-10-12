import os
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
    language = """Language: English"""
    
    with gr.Blocks(theme) as demo_chatbot:
        gr.Markdown(title_demo)
        gr.Markdown(title_paper)
        gr.Markdown(language)
        
        with gr.Row():
            with gr.Column(scale=3):
                start_btn = gr.Button("Start", variant="primary", interactive=True)
                clear_btn = gr.Button("Clear Context", interactive=False)
                
                
                with gr.Accordion("Text Generation Settings"):
                    do_sample = gr.Radio(["True", "False"],
                                         value="True",
                                         interactive=True,
                                         label="whether to do sample")
                    num_beams = gr.Slider(minimum=1, maximum=10, step=1,
                                          value=1,
                                          interactive=True,
                                          label="beam search size")
                    top_p = gr.Slider(minimum=0, maximum=1, step=0.01,
                                      value=0.7,
                                      interactive=True,
                                      label='"p" value of Top-P')
                    top_k = gr.Slider(minimum=1, maximum=50, step=1,
                                      value=20,
                                      interactive=True,
                                      label='"k" value of Top-K')
                    
                with gr.Accordion("Image Generation Settings"):
                    num_inference_steps = gr.Slider(minimum=20, maximum=70, step=1,
                                                    value=50,
                                                    interactive=True,
                                                    label="inference steps")
                    guidance_scale = gr.Slider(minimum=5, maximum=10, step=0.1,
                                               value=7.5,
                                               interactive=True,
                                               label="guidance scale")
                    
            with gr.Column(scale=7):
                chat_state = gr.State([])  # message, context_for_rmp, context_for_tdrg
                chatbot = gr.Chatbot(label="TIGER", height=700, avatar_images=((os.path.join(os.path.dirname(__file__), 'demo/user.png')), (os.path.join(os.path.dirname(__file__), "demo/bot.png"))))
                with gr.Row():
                    with gr.Column(scale=7):
                        text_input = gr.Textbox(label="User", placeholder="Please click the <Start or Restart> button to start chat!", interactive=False)
                    with gr.Column(scale=3):
                        undo_btn = gr.Button("undo", interactive=False)
                    
        with gr.Accordion("Prompt templates"):
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
        
        start_btn.click(agent.start_chat, [chat_state], [chatbot, chat_state, text_input, start_btn, clear_btn], queue=False)
        clear_btn.click(agent.restart_chat, [], [chatbot, chat_state, text_input, start_btn, clear_btn], queue=False)
        text_input.submit(
            fn=agent.respond, 
            inputs=[text_input, chatbot, chat_state, do_sample, num_beams, top_p, top_k, num_inference_steps, guidance_scale, human_words, human_prompt, others_prompt, negative_prompt], 
            outputs=[text_input, chatbot, chat_state, undo_btn]
        )
        undo_btn.click(agent.undo, [chatbot, chat_state], [text_input, chatbot, chat_state])
        
    demo_chatbot.launch(share=True, server_name="127.0.0.1", server_port=7860)
    demo_chatbot.queue()


if __name__ == "__main__":
    args = parse_args()

    main(args)

"""
Unfortunately, I didn't see any moonflowers. Can you show me what they look like?
That's a shame.
"""