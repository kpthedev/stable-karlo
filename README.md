# stable-karlo

![screenshot](https://user-images.githubusercontent.com/115115916/210150684-846dbc11-bd22-4cdf-90f9-23927158e1db.png)

A Streamlit app that combines [Karlo](https://github.com/kakaobrain/karlo) text-to-image generations with the [Stable-Diffusion v2](https://github.com/Stability-AI/stablediffusion) upscaler in a simple webUI.

**Implemented with:**
* Huggingface's [Diffusers](https://github.com/huggingface/diffusers)🧨
* [Streamlit](https://github.com/streamlit/streamlit)
* [xformers](https://github.com/facebookresearch/xformers) (optionally)

## Install
> Note that [xformers](https://github.com/facebookresearch/xformers) is not in the `requirements.txt`. Using it is optional, but I'd recommend it if you have a GPU with low memory. You can follow the instructions on their repo to get it set up in the python environment.

### Linux/MacOS
```bash
git clone https://github.com/kpthedev/stable-karlo.git
cd stable-karlo
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

### Windows
```bash
git clone https://github.com/kpthedev/stable-karlo.git
cd stable-karlo
python -m venv .env
.env\Scripts\activate
pip install -r requirements.txt
pip install --upgrade --force-reinstall torch --extra-index-url https://download.pytorch.org/whl/cu117
```

## Running
To run the app, make sure you are in the `stable-karlo` folder and have activated the environment, then run:

```bash
streamlit run app.py
```
This should open the webUI in your browser automatically.

> The very first time you run the app, it will download the models from Huggingface. This may take a while, depending on your internet speed—the models are around 18GB total.

### Memory Requirements
The Karlo model by itself requires a small amount of GPU memory (~8GB). However, the Stable Diffusion Upscaler requires significantly more VRAM. In the Upscaler settings, there are two methods of lowering the VRAM requirements:

* The first, is to downscale the Karlo image (originally 256x256 pixels) that is fed into the upscaler.
* The other is using xformers (which requires a working xformers installation).

## License
All the original code that I have written is licensed under a GPL license. The licenses for the respective model weights, are included in the repository.
