import os
import gradio as gr
import numpy as np
import spaces
import torch
import random
from PIL import Image
from typing import Iterable
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

colors.steel_blue = colors.Color(
    name="steel_blue",
    c50="#EBF3F8",
    c100="#D3E5F0",
    c200="#A8CCE1",
    c300="#7DB3D2",
    c400="#529AC3",
    c500="#4682B4",
    c600="#3E72A0",
    c700="#36638C",
    c800="#2E5378",
    c900="#264364",
    c950="#1E3450",
)

class SteelBlueTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.steel_blue,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
        )

steel_blue_theme = SteelBlueTheme()

from diffusers import FlowMatchEulerDiscreteScheduler
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3

dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509",
    transformer=QwenImageTransformer2DModel.from_pretrained(
        "linoyts/Qwen-Image-Edit-Rapid-AIO",
        subfolder='transformer',
        torch_dtype=dtype,
        device_map='cuda'
    ),
    torch_dtype=dtype
).to(device)

pipe.load_lora_weights("autoweeb/Qwen-Image-Edit-2509-Photo-to-Anime",
                       weight_name="Qwen-Image-Edit-2509-Photo-to-Anime_000001000.safetensors",
                       adapter_name="anime")
pipe.load_lora_weights("dx8152/Qwen-Edit-2509-Multiple-angles",
                       weight_name="镜头转换.safetensors",
                       adapter_name="multiple-angles")
pipe.load_lora_weights("dx8152/Qwen-Image-Edit-2509-Light_restoration",
                       weight_name="移除光影.safetensors",
                       adapter_name="light-restoration")
pipe.load_lora_weights("dx8152/Qwen-Image-Edit-2509-Relight",
                       weight_name="Qwen-Edit-Relight.safetensors",
                       adapter_name="relight")

pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
MAX_SEED = np.iinfo(np.int32).max

@spaces.GPU
def infer(
    input_image,
    prompt,
    lora_adapter,
    seed,
    randomize_seed,
    guidance_scale,
    steps,
    progress=gr.Progress(track_tqdm=True)
):
    if input_image is None:
        raise gr.Error("Please upload an image to edit.")

    if lora_adapter == "Photo-to-Anime":
        pipe.set_adapters(["anime"], adapter_weights=[1.0])
    elif lora_adapter == "Multiple-Angles":
        pipe.set_adapters(["multiple-angles"], adapter_weights=[1.0])
    elif lora_adapter == "Light-Restoration":
        pipe.set_adapters(["light-restoration"], adapter_weights=[1.0])
    elif lora_adapter == "Relight":
        pipe.set_adapters(["relight"], adapter_weights=[1.0])

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator(device=device).manual_seed(seed)
    negative_prompt = "worst quality, low quality, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry"

    original_image = input_image.convert("RGB")
    width, height = original_image.size

    result = pipe(
        image=original_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        generator=generator,
        true_cfg_scale=guidance_scale,
    ).images[0]

    return result, seed

@spaces.GPU
def infer_example(input_image, prompt, lora_adapter):
    input_pil = input_image.convert("RGB")
    guidance_scale = 1.0
    steps = 4
    result, seed = infer(input_pil, prompt, lora_adapter, 0, True, guidance_scale, steps)
    return result, seed


css="""
#col-container {
    margin: 0 auto;
    max-width: 960px;
}
#main-title h1 {font-size: 2.1em !important;}
"""

with gr.Blocks(css=css, theme=steel_blue_theme) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# **Qwen-Image-Edit-2509-LoRAs-Fast**", elem_id="main-title")
        gr.Markdown("Perform diverse image edits using specialized [LoRA](https://huggingface.co/models?other=base_model:adapter:Qwen/Qwen-Image-Edit-2509) adapters for the [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) model.")

        with gr.Row(equal_height=True):
            with gr.Column():
                input_image = gr.Image(label="Upload Image", type="pil")
                
                prompt = gr.Text(
                    label="Edit Prompt",
                    show_label=True,
                    placeholder="e.g., transform into anime..",
                )

                run_button = gr.Button("Run", variant="primary")

            with gr.Column():
                output_image = gr.Image(label="Output Image", interactive=False, format="png", height=290)
                
                with gr.Row():
                    lora_adapter = gr.Dropdown(
                        label="Choose Editing Style",
                        choices=["Photo-to-Anime", "Multiple-Angles", "Light-Restoration", "Relight"],
                        value="Photo-to-Anime"
                    )
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                    randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                    guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=10.0, step=0.1, value=1.0)
                    steps = gr.Slider(label="Inference Steps", minimum=1, maximum=50, step=1, value=4)
        
        gr.Examples(
            examples=[
                ["examples/1.jpg", "Transform into anime.", "Photo-to-Anime"],
                ["examples/5.jpg", "Remove shadows and relight the image using soft lighting.", "Light-Restoration"],
                ["examples/4.jpg", "Use a subtle golden-hour filter with smooth light diffusion.", "Relight"],
                ["examples/2.jpeg", "Rotate the camera 45 degrees to the left.", "Multiple-Angles"],
                ["examples/2.jpeg", "Switch the camera to a top-down right corner view.", "Multiple-Angles"],
                ["examples/6.jpg", "Switch the camera to a bottom-up view.", "Multiple-Angles"],
                ["examples/6.jpg", "Rotate the camera 180 degrees upside down.", "Multiple-Angles"],
                ["examples/4.jpg", "Rotate the camera 45 degrees to the right.", "Multiple-Angles"],
                ["examples/4.jpg", "Switch the camera to a top-down view.", "Multiple-Angles"],
                ["examples/4.jpg", "Switch the camera to a wide-angle lens.", "Multiple-Angles"],
            ],
            inputs=[input_image, prompt, lora_adapter],
            outputs=[output_image, seed],
            fn=infer_example,
            cache_examples=False,
            label="Examples"
        )

    run_button.click(
        fn=infer,
        inputs=[input_image, prompt, lora_adapter, seed, randomize_seed, guidance_scale, steps],
        outputs=[output_image, seed]
    )
    
demo.launch(mcp_server=True, ssr_mode=False, show_error=True)