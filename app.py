#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  app.py
#
#  Copyright 2023 KP
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
from models.generate import generate, upscale


def main():
    st.set_page_config(
        page_title="stable-karlo",
        page_icon="🖼️",
        layout="wide",
        menu_items={
            "Get Help": "https://github.com/kpthedev/stable-karlo",
            "Report a bug": "https://github.com/kpthedev/stable-karlo/issues",
            "About": "# stable-karlo 🖼️ \n Upscaling Karlo text-to-image generation using Stable Diffusion v2.",
        },
    )

    st.write("# stable-karlo 🖼️")
    col_left, col_right = st.columns(2)
    with col_left:
        st.info(
            "The first time you run this app, it will take some time to download all the models.",
            icon="ℹ️",
        )

        prompt = st.text_area("Prompt (77 words max)", key="karlo-prompt")
        n_images = st.slider("Number of images", 0, 8, 1)
        up = st.checkbox("Use Stable-Diffusion to upscale", value=True)

        with st.expander("Karlo Settings"):
            n_prior = st.slider("Number of prior steps", 0, 100, 25)
            n_decoder = st.slider("Number of decoder steps", 0, 100, 25)
            n_super_res = st.slider("Number of super res steps", 0, 100, 7)
            cfg_prior = st.slider(
                "Prior guidance scale", 0.0, 20.0, 4.0, step=0.1, format="%.1f"
            )
            cfg_decoder = st.slider(
                "Decoder guidance scale", 0.0, 20.0, 4.0, step=0.1, format="%.1f"
            )

            st.write("**Memory Optimizations:**")
            cpu_karlo = st.checkbox(
                "Use CPU offloading (significanlty reduces VRAM usage, but is much slower)",
                value=False,
                key="karlo-cpu",
            )

        with st.expander("Stable-Diffusion Upscaler Settings"):
            up_prompt = st.text_area("Prompt", prompt, key="sd-prompt")
            up_neg_prompt = st.text_area("Negative prompt")
            up_n_steps = st.slider("Number of steps", 0, 200, 50)
            up_cfg = st.slider(
                "Guidance scale", 1.01, 20.0, 7.5, step=0.1, format="%.1f"
            )
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

            st.write("**Memory Optimizations:**")
            cpu_sd = st.checkbox(
                "Use CPU offloading (significanlty reduces VRAM usage, but is much slower)",
                value=False,
                key="sd-cpu",
            )
            xfm_sd = st.checkbox(
                "Use xformers (requires xformers installed)",
                value=False,
                key="sd-xfm",
            )

        if st.button("Generate"):
            with st.spinner("Generating..."):
                images = generate(
                    cpu=cpu_karlo,
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
                    with st.spinner("Upscaling..."):
                        images_up = upscale(
                            cpu=cpu_sd,
                            xfm=xfm_sd,
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
        "\n * If you're getting `OutOfMemory` errors, try using the VRAM optimizations in each of the model settings.",
        '\n * If the "Use Stable-Diffusion to upscale" is not checked, the output is just from the Karlo model.',
    )


if __name__ == "__main__":
    main()
