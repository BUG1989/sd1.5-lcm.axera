from typing import List, Union
import numpy as np
import os
import tarfile
import onnxruntime
import torch
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel, PreTrainedTokenizer, CLIPTextModelWithProjection
from diffusers import UNet2DConditionModel, DiffusionPipeline, LCMScheduler, AutoencoderKL


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


def get_embeds(prompt = "Portrait of a pretty girl", ):
    tokenizer = CLIPTokenizer.from_pretrained("Lykon/dreamshaper-7/tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("Lykon/dreamshaper-7/text_encoder",
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

    timesteps = np.array([999, 759, 499, 259]).astype(np.int64)
    alphas_cumprod, final_alphas_cumprod, self_timesteps = get_alphas_cumprod()
    unet_session_main = onnxruntime.InferenceSession("output_onnx/unet_sim_cut.onnx")
    time_input = np.load("output_onnx/time_input.npy")
    
    os.makedirs("calib_data_unet", exist_ok=True)
    os.makedirs("calib_data_vae", exist_ok=True)
    
    calib_tarfile_unet = tarfile.open(f"calib_data_unet/data.tar", "w")
    calib_tarfile_vae = tarfile.open(f"calib_data_vae/data.tar", "w")

    prompts = ['Self-portrait oil painting, a beautiful cyborg with golden hair, 8k',
               'ultra close-up color photo portrait of rainbow owl with deer horns in the woods',
               'woman with a blue headscarf and a blue sweaterp',
               'Kung Fu Panda',
               'a majestic snowy mountain range under a clear night sky filled with sparkling stars and a bright Milky Way',
               'Portrait of a pretty girl',
               'A brain riding a rocketship heading towards the moon',
               'a robot with rockets blasting off from its feet',
               'A dragon fruit wearing karate belt in the snow.',
               'An extremely angry bird.',
               'A photo of a Corgi dog riding a bike in Times Square. It is wearing sunglasses and a beach hat.',
               'A small cactus wearing a straw hat and neon sunglasses in the Sahara desert.',
               'A blue jay standing on a large basket of rainbow macarons.',
               'A giant cobra snake on a farm. The snake is made out of corn.',
               'Teddy bears swimming at the Olympics 400m Butterfly event.',
               'A strawberry mug filled with white sesame seeds. The mug is floating in a dark chocolate sea.',
               'A cute corgi lives in a house made out of sushi.',
               'A golden Retriever dog wearing a blue checkered beret and red dotted turtle neck',
               'A dog looking curiously in the mirror',
               'A bald eagle made of chocolate powder, mango, and whipped cream.']
    for p, prompt in enumerate(prompts):
        prompt_embeds_npy = get_embeds(prompt)
        prompt_name = prompt.replace(" ", "_")
        latent = torch.randn([1, 4, 64, 64], generator=None, device="cpu", dtype=torch.float32, layout=torch.strided).detach().numpy()
        print(p, prompt)
        for i, timestep in enumerate(timesteps):
            print(i, timestep)
            noise_pred = unet_session_main.run(None,
                                               {"sample": latent,
                                                "/down_blocks.0/resnets.0/act_1/Mul_output_0": np.expand_dims(time_input[i], axis=0),
                                                "encoder_hidden_states": prompt_embeds_npy})[0]

            calib_data = {}
            calib_data["sample"] = latent
            calib_data["/down_blocks.0/resnets.0/act_1/Mul_output_0"] = np.expand_dims(time_input[i], axis=0)
            calib_data["encoder_hidden_states"] = prompt_embeds_npy
            np.save(f"calib_data_unet/data_{p}_{i}.npy", calib_data)
            calib_tarfile_unet.add(f"calib_data_unet/data_{p}_{i}.npy")

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

        latent = latent / 0.18215
        calib_data = {}
        calib_data["x"] = latent
        np.save(f"calib_data_vae/data_{p}.npy", calib_data)
        calib_tarfile_vae.add(f"calib_data_vae/data_{p}.npy")
        
    calib_tarfile_unet.close()
    calib_tarfile_vae.close()
