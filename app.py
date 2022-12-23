#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  app.py
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

import streamlit as st
from model.generate import generate, upscale


def main():
    st.set_page_config(layout="wide", page_title="karlo-ui")
    st.write("# stable-karlo")
    st.info(
        "The first time you run this app, it will take some time to download all the models.",
        icon="ℹ️",
    )

    col_left, col_right = st.columns(2)
    with col_left:
        prompt = st.text_area("Prompt", key="karlo-prompt")
        n_images = st.slider("Number of images", 0, 8, 1)
        up = st.checkbox("Use SD-v2 to upscale", value=True)

        with st.expander("Karlo Settings"):
            n_prior = st.slider("Number of prior steps", 0, 100, 25)
            n_decoder = st.slider("Number of decoder steps", 0, 100, 25)
            n_super_res = st.slider("Number of super res steps", 0, 100, 7)
            cfg_prior = st.slider("Prior guidance scale", 0.0, 20.0, 4.0)
            cfg_decoder = st.slider("Decoder guidance scale", 0.0, 20.0, 4.0)

        with st.expander("Stable-Diffusion Upscaler Settings"):
            up_prompt = st.text_area("Prompt", prompt, key="sd-prompt")
            up_neg_prompt = st.text_area("Negative prompt")
            up_n_steps = st.slider("Number of steps", 0, 200, 50)
            up_cfg = st.slider("Guidance scale", 1.01, 20.0, 7.5)
            up_downscale = st.slider(
                "Downscale input image (if you're running out of VRAM)",
                64,
                256,
                256,
                format="%ipx",
            )
            up_scheduler = st.radio(
                "Scheduler", ("DDIM", "LMS", "Euler"), horizontal=True
            )
            show_original = st.checkbox("Show original images", value=False)
            xfm = st.checkbox("Use xformers", value=False)

        if st.button("Generate"):
            images = generate(
                prompt=prompt,
                n_images=n_images,
                n_prior=n_prior,
                n_decoder=n_decoder,
                n_super_res=n_super_res,
                cfg_prior=cfg_prior,
                cfg_decoder=cfg_decoder,
            )

            with col_right:
                if up:
                    if show_original:
                        st.image(images)
                    images_up = upscale(
                        xfm_on=xfm,
                        downscale=up_downscale,
                        scheduler=up_scheduler,
                        prompt=up_prompt,
                        neg_prompt=up_neg_prompt,
                        images=images,
                        n_steps=up_n_steps,
                        cfg=up_cfg,
                    )
                    st.image(images_up)
                else:
                    st.image(images)

    st.write(
        "**Notes:**",
        "\n * If you're running out of VRAM, the Upscaler Settings has downscaling and xformers options.",
        "\n * If the *Use SD-v2 to upscale* is not checked, the output is just from Karlo.",
    )


if __name__ == "__main__":
    main()
