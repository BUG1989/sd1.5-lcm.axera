from typing import List, Union
import numpy as np
# import onnxruntime
import torch
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel, PreTrainedTokenizer, CLIPTextModelWithProjection
# from diffusers import UNet2DConditionModel, DiffusionPipeline, LCMScheduler, AutoencoderKL

from axengine import InferenceSession
import time
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        prog="StableDiffusion",
        description="Generate picture with the input prompt"
    )
    parser.add_argument("--prompt", "-p", type=str, required=False, default="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k", help="the input text prompt")
    parser.add_argument("--text_model_dir", "-e", type=str, required=False, default="./models/", help="the dir of text encoder and tokenizer files")
    parser.add_argument("--unet_model", "-u", type=str, required=False, default="./models/unet.axmodel", help="the dir of unet.axmodel")
    parser.add_argument("--vae_model", "-v", type=str, required=False, default="./models/vae.axmodel", help="the dir of vae.axmodel")
    parser.add_argument("--time_input", "-t", type=str, required=False, default="./models/time_input.npy", help="the dir of time input file")
    parser.add_argument("--save_dir", "-s", type=str, required=False, default="./lcm_lora_sdv1_5_axmodel.png", help="the save dir of the output image")
    return parser.parse_args()

def maybe_convert_prompt(prompt: Union[str, List[str]], tokenizer: "PreTrainedTokenizer"):  # noqa: F821
    if not isinstance(prompt, List):
        prompts = [prompt]
    else:
        prompts = prompt

    prompts = [_maybe_convert_prompt(p, tokenizer) for p in prompts]

    if not isinstance(prompt, List):
        return prompts[0]

    return prompts


def _maybe_convert_prompt(prompt: str, tokenizer: "PreTrainedTokenizer"):  # noqa: F821
    tokens = tokenizer.tokenize(prompt)
    unique_tokens = set(tokens)
    for token in unique_tokens:
        if token in tokenizer.added_tokens_encoder:
            replacement = token
            i = 1
            while f"{token}_{i}" in tokenizer.added_tokens_encoder:
                replacement += f" {token}_{i}"
                i += 1

            prompt = prompt.replace(token, replacement)

    return prompt


def get_embeds(prompt = "Portrait of a pretty girl", tokenizer_dir = "./models/tokenizer", text_encoder_dir = "./models/text_encoder"):
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_dir)
    text_encoder = CLIPTextModel.from_pretrained(text_encoder_dir,
                                                 torch_dtype=torch.float32,
                                                 variant="fp16")
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to("cpu"), attention_mask=None)

    prompt_embeds_npy = prompt_embeds[0].detach().numpy()
    return prompt_embeds_npy


def get_alphas_cumprod():
    betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=torch.float32) ** 2
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).detach().numpy()
    final_alphas_cumprod = alphas_cumprod[0]
    self_timesteps = np.arange(0, 1000)[::-1].copy().astype(np.int64)
    return alphas_cumprod, final_alphas_cumprod, self_timesteps


if __name__ == '__main__':
    args = get_args()
    prompt = args.prompt
    tokenizer_dir = args.text_model_dir + 'tokenizer'
    text_encoder_dir = args.text_model_dir + 'text_encoder'
    unet_model = args.unet_model
    vae_model = args.vae_model
    time_input = args.time_input
    save_dir = args.save_dir

    print(f"prompt: {prompt}")
    print(f"text_tokenizer: {tokenizer_dir}")
    print(f"text_encoder: {text_encoder_dir}")
    print(f"unet_model: {unet_model}")
    print(f"vae_model: {vae_model}")
    print(f"time_input: {time_input}")
    print(f"save_dir: {save_dir}")

    timesteps = np.array([999, 759, 499, 259]).astype(np.int64)
    
    # text encoder
    start = time.time()    
    # prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
    prompt_embeds_npy = get_embeds(prompt, tokenizer_dir, text_encoder_dir)
    print(f"text encoder take {1000 * (time.time() - start)}ms")
    
    prompt_name = prompt.replace(" ", "_")
    latents_shape = [1, 4, 64, 64]
    latent = torch.randn(latents_shape, generator=None, device="cpu", dtype=torch.float32,
                         layout=torch.strided).detach().numpy()
    
    alphas_cumprod, final_alphas_cumprod, self_timesteps = get_alphas_cumprod()
    
    # load unet model and vae model
    start = time.time()    
    unet_session_main = InferenceSession.load_from_model(unet_model)
    vae_decoder = InferenceSession.load_from_model(vae_model)
    print(f"load models take {1000 * (time.time() - start)}ms")
    
    # load time input file
    time_input = np.load(time_input)
    
    # unet inference loop
    unet_loop_start = time.time()    
    for i, timestep in enumerate(timesteps):
        # print(i, timestep)
        
        unet_start = time.time()
        noise_pred = unet_session_main.run({"sample": latent, \
                                            "/down_blocks.0/resnets.0/act_1/Mul_output_0": np.expand_dims(time_input[i], axis=0), \
                                            "encoder_hidden_states": prompt_embeds_npy})['5771']
        print(f"unet once take {1000 * (time.time() - unet_start)}ms")

        sample = latent
        model_output = noise_pred
        if i < 3:
            prev_timestep = timesteps[i + 1]
        else:
            prev_timestep = timestep

        alpha_prod_t = alphas_cumprod[timestep]
        alpha_prod_t_prev = alphas_cumprod[prev_timestep] if prev_timestep >= 0 else final_alphas_cumprod

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 3. Get scalings for boundary conditions
        scaled_timestep = timestep * 10
        c_skip = 0.5 ** 2 / (scaled_timestep ** 2 + 0.5 ** 2)
        c_out = scaled_timestep / (scaled_timestep ** 2 + 0.5 ** 2) ** 0.5
        predicted_original_sample = (sample - (beta_prod_t ** 0.5) * model_output) / (alpha_prod_t ** 0.5)

        denoised = c_out * predicted_original_sample + c_skip * sample

        if i != 3:
            noise = torch.randn(model_output.shape, generator=None, device="cpu", dtype=torch.float32,
                                layout=torch.strided).to("cpu").detach().numpy()
            prev_sample = (alpha_prod_t_prev ** 0.5) * denoised + (beta_prod_t_prev ** 0.5) * noise
        else:
            prev_sample = denoised

        latent = prev_sample

    print(f"unet loop take {1000 * (time.time() - unet_loop_start)}ms")

    # vae inference
    vae_start = time.time()    
    latent = latent / 0.18215
    image = vae_decoder.run({"x": latent})['784']
    print(f"vae inference take {1000 * (time.time() - vae_start)}ms")
    
    # save result
    save_start = time.time() 
    image = np.transpose(image, (0, 2, 3, 1)).squeeze(axis=0)
    image_denorm = np.clip(image / 2 + 0.5, 0, 1)
    image = (image_denorm * 255).round().astype("uint8")
    pil_image = Image.fromarray(image[:, :, :3])
    pil_image.save(save_dir)
    print(f"save image take {1000 * (time.time() - vae_start)}ms")
