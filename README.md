# stable-karlo

![screenshot](https://user-images.githubusercontent.com/115115916/209260866-fb3a6cf2-060e-46b7-b89f-778db1d14e4d.jpg)

A Streamlit app that combines [Karlo](https://github.com/kakaobrain/karlo) text-to-image generations with the [Stable-Diffusion v2](https://github.com/Stability-AI/stablediffusion) upscaler in a simple webUI.

**Implemented with:**
* Huggingface's [Diffusers](https://github.com/huggingface/diffusers)🧨
* [Streamlit](https://github.com/streamlit/streamlit)
* [xformers](https://github.com/facebookresearch/xformers) (optionally)

## Install
```bash
git clone https://github.com/kpthedev/stable-karlo.git
cd stable-karlo
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

Note that [xformers](https://github.com/facebookresearch/xformers) is not in the `requirements.txt`; however, it's highly recommended that you follow the instructions on their repo to get it set up in the python environment.

## Running
To run the app, make sure you are in the `stable-karlo` folder and have activated the environment, then run:

```bash
streamlit run app.py
```
This should open the webUI in your browser automatically.

> The very first time you run the app, it will download the models from Huggingface. This may take a while, depending on your internet speed—the models are around 18G total.

### VRAM Requirements
The Karlo model requires a moderate amount of VRAM; however, the Stable Diffusion Upscaler requires a significant amount of GPU memory. In the Upscaler settings, there are two methods of lowering VRAM requirements:

* The first, is to downscale the Karlo image (256x256 pixels) that is fed into the upscaler.
* The other is using xformers (which requires a working xformers installation).

## License
All the original code that I have written is licensed under a GPL license. The licenses for the respective model weights, are included in the repository.
