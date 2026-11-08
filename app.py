"""SDXL on Modal, L4 GPU, diffusers. Based on Modal's official SDXL example,
with gpu changed from H100 to L4 and the same prompt used for the Colab test.
Run: modal run app.py
"""
import io
import os

import modal

app = modal.App("sdxl-l4")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "diffusers==0.31.0",
        "transformers==4.46.2",
        "accelerate==1.1.1",
        "safetensors==0.4.5",
        "torch==2.5.1",
    )
    .env({"HF_HUB_CACHE": "/cache"})
)

volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)


@app.cls(gpu="L4", volumes={"/cache": volume}, timeout=600)
class SDXLModel:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import DiffusionPipeline

        self.pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        self.pipe.to("cuda")

    @modal.method()
    def generate(self, prompt: str) -> bytes:
        image = self.pipe(
            prompt=prompt, num_inference_steps=30, guidance_scale=7.5
        ).images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()


@app.local_entrypoint()
def main():
    prompt = (
        "a cozy mountain cabin at sunset, snow-covered pine trees, "
        "warm lights in windows, photorealistic, highly detailed"
    )
    data = SDXLModel().generate.remote(prompt)
    out = os.path.join(os.path.dirname(__file__), "sdxl_test_modal.png")
    with open(out, "wb") as f:
        f.write(data)
    print(f"Saved {out} ({len(data)} bytes)")
