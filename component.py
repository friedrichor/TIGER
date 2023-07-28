import gradio as gr




def decoding_setting(stratey: str, num_beams: gr.Slider, p_value: gr.Slider, k_value: gr.Slider):
    print(stratey)
    if stratey == "Beam":
        return gr.Slider.update(visible=True), gr.Slider.update(visible=False), gr.Slider.update(visible=False)
    elif stratey == "Top-P":
        return gr.Slider.update(visible=False), gr.Slider.update(visible=True), gr.Slider.update(visible=False)
    elif stratey == "Top-K":
        return gr.Slider.update(visible=False), gr.Slider.update(visible=False), gr.Slider.update(visible=True)