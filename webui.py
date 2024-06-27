import argparse
import time

import gradio as gr
import numpy as np
import os

import scipy
import torch
import ChatTTS

current_dir = os.path.dirname(os.path.realpath(__file__))

model_path = f"{current_dir}/models/chatTTS"

print(f"loading ChatTTS model from '{model_path}")
chat = ChatTTS.Chat()

chat.load_models(
    vocos_config_path=f"{model_path}/config/vocos.yaml",
    dvae_config_path=f"{model_path}/config/dvae.yaml",
    gpt_config_path=f"{model_path}/config/gpt.yaml",
    decoder_config_path=f"{model_path}/config/decoder.yaml",

    vocos_ckpt_path=f"{model_path}/asset/Vocos.pt",
    dvae_ckpt_path=f"{model_path}/asset/DVAE.pt",
    gpt_ckpt_path=f"{model_path}/asset/GPT.pt",
    decoder_ckpt_path=f"{model_path}/asset/Decoder.pt",
    tokenizer_path=f"{model_path}/asset/tokenizer.pt",
)
std, mean = torch.load(f"{model_path}/asset/spk_stat.pt").chunk(2)


def generate_audio(text, refine_text_flag, temperature, top_p, top_k, oral_slider, laugh_slider, break_slider):
    print(f"generating audio for text: {text}")
    if refine_text_flag:
        rand_spk = torch.randn(768) * std + mean
        print("rand_spk："+str(rand_spk))
        params_infer_code = {
            'spk_emb': rand_spk,
            'temperature': temperature,
            'top_P': top_p,
            'top_K': top_k,
        }
        print(f"temperature: {temperature}, top_p: {top_p}, top_k: {top_k}")
        params_refine_text = {
            'prompt': f"[oral_{oral_slider}][laugh_{laugh_slider}][break_{break_slider}]"
        }
        print(f"params_refine_text: {params_refine_text}")
        wav = chat.infer(text, params_refine_text=params_refine_text, params_infer_code=params_infer_code)
    else:
        wav = chat.infer(text)

    output_dir = f"{current_dir}/output"
    if os.name == 'nt':
        output_dir = f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filename = f"{output_dir}/{time.time()}.wav"
    scipy.io.wavfile.write(filename=filename, rate=24_000, data=wav[0].T)

    audio_data = np.array(wav[0]).flatten()
    sample_rate = 24000
    text_data = text[0] if isinstance(text, list) else text
    return [(sample_rate, audio_data), text_data]



# [oral_0-9][laugh_0-2][break_0-7]
def main():
    with gr.Blocks() as demo:
        gr.Markdown("# ChatTTS 语音合成演示")
        gr.Markdown("> by 网旭哈瑞")
        default_text = "你眼中的自己不是你，别人眼中的你也不是你，你眼中的别人才是你自己。"
        text_input = gr.Textbox(label="输入要合成的文字", lines=4, placeholder="输入要合成的文字...",
                                value=default_text)

        def update_sliders(refine_text):
            enabled = refine_text
            return [gr.update(interactive=enabled)] * 6

        with gr.Group():
            with gr.Row():
                refine_text_checkbox = gr.Checkbox(label="细粒度控制", value=False)
            with gr.Row():
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(" 生成参数")
                        temperature_slider = gr.Slider(minimum=0.00001, maximum=1.0, step=0.00001, value=0.3,
                                                       label="temperature")
                        top_p_slider = gr.Slider(minimum=0.1, maximum=0.9, step=0.05, value=0.7, label="topP")
                        top_k_slider = gr.Slider(minimum=1, maximum=20, step=1, value=20, label="topK")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown(" 韵律特征")
                        oral_slider = gr.Slider(minimum=0, maximum=9, step=1, value=2, label="插入词（oral）强度")
                        laugh_slider = gr.Slider(minimum=0, maximum=2, step=1, value=0, label="笑声（laugh）强度")
                        break_slider = gr.Slider(minimum=0, maximum=7, step=1, value=6, label="停顿（break）强度")

        refine_text_checkbox.change(
            update_sliders,
            inputs=[refine_text_checkbox],
            outputs=[temperature_slider, top_p_slider, top_k_slider, oral_slider, laugh_slider, break_slider]
        )

        generate_button = gr.Button("生成语音", variant="primary")

        text_output = gr.Textbox(label="文字", interactive=False, visible=False)
        audio_output = gr.Audio(label="音频")

        generate_button.click(generate_audio,
                              inputs=[text_input, refine_text_checkbox, temperature_slider, top_p_slider, top_k_slider,
                                      oral_slider, laugh_slider, break_slider],
                              outputs=[audio_output, text_output])

    parser = argparse.ArgumentParser(description='ChatTTS demo Launch')
    parser.add_argument('--server_name', type=str, default='0.0.0.0', help='Server name')
    parser.add_argument('--server_port', type=int, default=8845, help='Server port')
    args = parser.parse_args()

    demo.launch(server_name=args.server_name, server_port=args.server_port, inbrowser=True)


if __name__ == '__main__':
    main()
