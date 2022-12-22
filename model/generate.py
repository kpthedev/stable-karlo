#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  generate.py
#
#  Copyright 2022 KP
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#

import torch
import streamlit as st
from diffusers import (
    UnCLIPPipeline,
    StableDiffusionUpscalePipeline,
    DDIMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
)
from PIL import Image


@st.cache(allow_output_mutation=True)
def make_pipe():
    pipe = UnCLIPPipeline.from_pretrained(
        "kakaobrain/karlo-v1-alpha", torch_dtype=torch.float16
    )
    return pipe.to("cuda")


@st.cache(allow_output_mutation=True)
def make_pipe_up(scheduler):
    if scheduler == "Euler":
        scheduler = EulerDiscreteScheduler.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler", subfolder="scheduler"
        )
    elif scheduler == "LMS":
        scheduler = LMSDiscreteScheduler.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler", subfolder="scheduler"
        )
    else:
        scheduler = DDIMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler", subfolder="scheduler"
        )

    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler",
        scheduler=scheduler,
        torch_dtype=torch.float16,
    )
    return pipe.to("cuda")


def generate(prompt, n_images, n_prior, n_decoder, n_super_res, cfg_prior, cfg_decoder):
    pipe = make_pipe()
    torch.cuda.empty_cache()
    with torch.autocast("cuda"):
        images = pipe(
            prompt=prompt,
            num_images_per_prompt=n_images,
            prior_num_inference_steps=n_prior,
            decoder_num_inference_steps=n_decoder,
            super_res_num_inference_steps=n_super_res,
            prior_guidance_scale=cfg_prior,
            decoder_guidance_scale=cfg_decoder,
        ).images
    return images


def upscale(downscale, scheduler, prompt, neg_prompt, images, n_steps, cfg):

    batch_prompt = [prompt] * len(images)
    batch_neg_prompt = [neg_prompt] * len(images)
    for i in range(len(images)):
        images[i] = images[i].resize((downscale, downscale))

    pipe = make_pipe_up(scheduler)
    pipe.enable_attention_slicing()
    torch.cuda.empty_cache()
    images = pipe(
        image=images,
        prompt=batch_prompt,
        negative_prompt=batch_neg_prompt,
        num_inference_steps=n_steps,
        guidance_scale=cfg,
    ).images
    return images
