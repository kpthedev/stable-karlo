# stable-karlo

![screenshot](https://user-images.githubusercontent.com/115115916/210285673-833ee286-c1a0-4d9d-a1e7-92991b9eb2f6.png)

A Streamlit app that combines [Karlo](https://github.com/kakaobrain/karlo) text-to-image generations with the [Stable-Diffusion v2](https://github.com/Stability-AI/stablediffusion) upscaler in a simple webUI.

<br/>

> ### Now available on Google Colab:
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kpthedev/stable-karlo-colab/blob/main/stable_karlo_colab.ipynb)

<br/>

**Built with:**
* [Huggingface Diffusers](https://github.com/huggingface/diffusers)🧨
* [Streamlit](https://github.com/streamlit/streamlit)
* [xformers](https://github.com/facebookresearch/xformers)

## Install (Local)

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
In the settings of each model, there are options for lowering the VRAM requirements:

* Both model settings have a **Use CPU offloading** option, which will substantially lower the VRAM usage.
* The Upscaler has two other methods to lower the VRAM usage:
  * **Downscale input image** - takes the Karlo image that is fed into the upscaler and downscales it to the specified size
  * **Use xformers** - uses xformers' efficent memory attention to lower the VRAM usage (this option requires a working xformers installation)
  
  ---
    
  | Model | Optimizations | VRAM Usage |
  |--------|---------------|------------|
  | Karlo | none | 10GB |
  | Karlo | CPU-offloading | 7GB |
  
  | Model | Optimizations | VRAM Usage |
  |--------|---------------|------------|
  | Karlo + Upscale | none | >24GB |
  | Karlo + Upscale | Downscale to < 190px | 12GB |
  | Karlo + Upscale | xformers | 15GB |
  | Karlo + Upscale | CPU-offloading + xformers | 15GB |
  | Karlo + Upscale | CPU-offloading + Downscale to < 190px | 12GB |
  | Karlo + Upscale | CPU-offloading + xformers + Downscale to < 190px | 10GB |

## License
All the original code that I have written is licensed under a GPL license. The licenses for the respective model weights, are included in the repository.

## Changelog
* Dec 22, 2022 - Inital release
* Jan 02, 2023 - Add CPU offloading
* Jan 18, 2023 - Add Google Colab support
