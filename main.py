import streamlit as st
import pandas as pd

st.set_page_config(page_title="AI4SocialGood By Bill L.", layout="wide")

st.title("AI4SocialGood: Semantic and Topological Modeling of Autism Support Social Communities")

st.write("""
## Project Overview
This project presents an empirical AI-driven network analysis of three leading autism-related organizations — AutismBC, ACAR, and ASF — by examining their follower and following networks on X (formerly Twitter). Using automated web scraping and machine learning methods, we construct and decode each organization's social structure. We identify overlap across the organizations, evaluate structural complexity using node2vec embeddings, and explore the semantic composition of their extended communities using bio descriptions. Despite limitations due to recent X API restrictions, our results shed light on the social and semantic reach of these organizations, revealing low structural complexity and limited interconnectedness. We propose this approach as a scalable framework for future network-based evaluations of digital advocacy ecosystems.

""")