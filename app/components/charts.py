"""
Charts.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st


def plot_probability(probability):

    fig, ax = plt.subplots(figsize=(6, 1.8))

    ax.barh(
        ["PD"],
        [probability],
    )

    ax.set_xlim(0, 1)

    st.pyplot(fig)