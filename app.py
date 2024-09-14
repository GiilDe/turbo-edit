from __future__ import annotations
import argparse

import gradio as gr
import spaces

from main import load_pipe, run as run_model


DESCRIPTION = """# Turbo Edit
"""

parser = argparse.ArgumentParser()
parser.add_argument("--cache_dir", type=str, default=None)
args = parser.parse_args()

pipeline = load_pipe(True, args.cache_dir)


@spaces.GPU
def main_pipeline(
    input_image: str,
    src_prompt: str,
    tgt_prompt: str,
    seed: int,
    w1: float,
):
    res_image = run_model(input_image, src_prompt, tgt_prompt, seed, w1, 4, pipeline)

    return res_image


with gr.Blocks(css="app/style.css") as demo:
    gr.Markdown(DESCRIPTION)

    gr.HTML(
        """<a href="https://huggingface.co/spaces/turboedit/turbo_edit?duplicate=true">
        <img src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>Duplicate the Space to run privately without waiting in queue"""
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Input image", type="filepath", height=512, width=512
            )
            src_prompt = gr.Text(
                label="Source Prompt",
                max_lines=1,
                placeholder="Source Prompt",
            )
            tgt_prompt = gr.Text(
                label="Target Prompt",
                max_lines=1,
                placeholder="Target Prompt",
            )
            with gr.Accordion("Advanced Options", open=False):
                seed = gr.Slider(
                    label="seed", minimum=0, maximum=16 * 1024, value=7865, step=1
                )
                w1 = gr.Slider(
                    label="w", minimum=1.0, maximum=3.0, value=1.5, step=0.05
                )

            run_button = gr.Button("Edit")
        with gr.Column():
            result = gr.Image(label="Result", type="pil", height=512, width=512)

            examples = [
                [
                    "examples_demo/1.jpeg",  # input_image
                    "a dreamy cat sleeping on a floating leaf",  # src_prompt
                    "a dreamy bear sleeping on a floating leaf",  # tgt_prompt
                    7,  # seed
                    1.3,  # w1
                ],
                [
                    "examples_demo/2.jpeg",  # input_image
                    "A painting of a cat and a bunny surrounded by flowers",  # src_prompt
                    "a polygonal illustration of a cat and a bunny",  # tgt_prompt
                    2,  # seed
                    1.5,  # w1
                ],
                [
                    "examples_demo/3.jpg",  # input_image
                    "a chess pawn wearing a crown",  # src_prompt
                    "a chess pawn wearing a hat",  # tgt_prompt
                    2,  # seed
                    1.3,  # w1
                ],
            ]

            gr.Examples(
                examples=examples,
                inputs=[
                    input_image,
                    src_prompt,
                    tgt_prompt,
                    seed,
                    w1,
                ],
                outputs=[result],
                fn=main_pipeline,
                cache_examples=True,
            )

    inputs = [
        input_image,
        src_prompt,
        tgt_prompt,
        seed,
        w1,
    ]
    outputs = [result]
    run_button.click(fn=main_pipeline, inputs=inputs, outputs=outputs)

demo.queue(max_size=50).launch(share=False)
