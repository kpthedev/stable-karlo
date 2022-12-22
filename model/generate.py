#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  generate.py
#
#  Copyright 2022 KP <LOCALHOST>
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
from diffusers import UnCLIPPipeline


def make_pipe():
    pipe = UnCLIPPipeline.from_pretrained(
        "kakaobrain/karlo-v1-alpha", torch_dtype=torch.float16
    )
    return pipe.to("cuda")


def generate(prompt, n_images, n_prior, n_decoder, n_super_res, cfg_prior, cfg_decoder):
    pipe = make_pipe()
    images = pipe(
        prompt=prompt,
        num_images_per_prompt=n_images,
        prior_num_inference_steps=n_prior,
        decoder_num_inference_steps=n_decoder,
        super_res_num_inference_steps=n_super_res,
        prior_guidance_scale=cfg_prior,
        decoder_guidance_scale=cfg_decoder,
    )
    return images
