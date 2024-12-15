import argparse
import pathlib
import numpy as np
import onnx
import onnxsim
import torch
from diffusers import LCMScheduler, AutoPipelineForText2Image, AutoencoderKL

"""
test env:
    protobuf:3.20.3
    onnx:1.16.0
    onnxsim:0.4.36
    torch:2.1.2+cu121
    transformers:4.45.0
"""


def extract_by_hand(input_model):
    input_graph = input_model
    to_remove_node = []
    for node in input_graph.node:
        if (
            node.name.startswith("/time_proj")
            or node.name.startswith("/time_embedding")
            or node.name
            in [
                "/down_blocks.0/resnets.0/act_1/Sigmoid",
                "/down_blocks.0/resnets.0/act_1/Mul",
            ]
        ):
            to_remove_node.append(node)
        else:
            pass
    for node in to_remove_node:
        input_graph.node.remove(node)
    to_remove_input = []
    for input in input_graph.input:
        if input.name in ["t"]:
            to_remove_input.append(input)
    for input in to_remove_input:
        input_graph.input.remove(input)
    new_input = []
    for value_info in input_graph.value_info:
        if value_info.name == "/down_blocks.0/resnets.0/act_1/Mul_output_0":
            new_input.append(value_info)
    input_graph.input.extend(new_input)


def extract_unet(input_path, input_lora_path, output_path):
    pipe = AutoPipelineForText2Image.from_pretrained(
        input_path, torch_dtype=torch.float32, variant="fp16"
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    # load and fuse lcm lora
    pipe.load_lora_weights(
        str(pathlib.Path(input_lora_path) / "pytorch_lora_weights.safetensors")
    )
    pipe.fuse_lora()
    """
        extract unet
    """
    extract_unet = True
    if extract_unet:
        pipe.unet.eval()

        class UNETWrapper(torch.nn.Module):
            def __init__(self, unet):
                super().__init__()
                self.unet = unet

            def forward(self, sample=None, t=None, encoder_hidden_states=None):
                return self.unet.forward(sample, t, encoder_hidden_states)

        example_input = {
            "sample": torch.rand([1, 4, 64, 64], dtype=torch.float32),
            "t": torch.from_numpy(np.array(1, dtype=np.int64)),
            "encoder_hidden_states": torch.rand([1, 77, 768], dtype=torch.float32),
        }

        unet_path = pathlib.Path(output_path) / "unet"
        if not unet_path.exists():
            unet_path.mkdir()
        torch.onnx.export(
            UNETWrapper(pipe.unet),
            tuple(example_input.values()),
            str(pathlib.Path(output_path) / "unet" / "unet.onnx"),
            opset_version=17,
            verbose=False,
            input_names=list(example_input.keys()),
        )
        unet = onnx.load(str(pathlib.Path(output_path) / "unet" / "unet.onnx"))
        unet_sim, _ = onnxsim.simplify(unet)
        extract_by_hand(unet_sim.graph)
        onnx.save(
            unet_sim,
            str(pathlib.Path(output_path) / "unet_sim_cut.onnx"),
            save_as_external_data=True,
        )

    """
        precompute time embedding
    """
    time_input = np.zeros([4, 1280], dtype=np.float32)
    timesteps = np.array([999, 759, 499, 259]).astype(np.int64)
    for i, t in enumerate(timesteps):
        tt = torch.from_numpy(np.array([t])).to(torch.float32)
        sample = pipe.unet.time_proj(tt)
        res = pipe.unet.time_embedding(sample)
        res = torch.nn.functional.silu(res)
        res_npy = res.detach().numpy()[0]
        time_input[i, :] = res_npy
    np.save(str(pathlib.Path(output_path) / "time_input.npy"), time_input)


def extract_vae(input_path, output_path):
    vae = AutoencoderKL.from_pretrained(
        str(pathlib.Path(input_path) / "vae"), torch_dtype=torch.float32, variant="fp16"
    )
    dummy_input = torch.rand([1, 4, 64, 64], dtype=torch.float32)
    vae.eval()

    class VAEWrapper(torch.nn.Module):
        def __init__(self, conv_quant, decoder):
            super().__init__()
            self.conv_quant = conv_quant
            self.decoder = decoder

        def forward(self, sample=None):
            sample = self.conv_quant(sample)
            decoder = self.decoder(sample)
            return decoder

    vaewrapper = VAEWrapper(vae.post_quant_conv, vae.decoder)
    torch.onnx.export(
        vaewrapper,
        dummy_input,
        str(pathlib.Path(output_path) / "sd15_vae_decoder.onnx"),
        opset_version=17,
        verbose=False,
        input_names=["x"],
    )
    vae = onnx.load(str(pathlib.Path(output_path) / "sd15_vae_decoder.onnx"))
    vae_sim, _ = onnxsim.simplify(vae)
    onnx.save(vae_sim, str(pathlib.Path(output_path) / "sd15_vae_decoder_sim.onnx"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="unet extract")
    parser.add_argument("--input_path", required=True, help="download sd_15 path")
    parser.add_argument(
        "--input_lora_path", help="download lora weight path", required=True
    )
    parser.add_argument("--output_path", help="output path", required=True)

    args = parser.parse_args()

    extract_unet(args.input_path, args.input_lora_path, args.output_path)
    extract_vae(args.input_path, args.output_path)
    