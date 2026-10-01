"""AI-Powered Healthcare Risk Assessment Assistant: Streamlit entry point.

Run: streamlit run app.py

Navigation only. Each area is a page script in ui/; all model logic lives in src/.
"""
import streamlit as st

from ui import core

if __name__ == "__main__":
    st.set_page_config(page_title="AI-Powered Healthcare Risk Assessment Assistant",
                       page_icon=":material/monitor_heart:", layout="wide")
    pages = {
        "assess": st.Page("ui/assess.py", title="Assess", icon=":material/stethoscope:", default=True),
        "explain": st.Page("ui/explain.py", title="Explain", icon=":material/insights:"),
        "explore": st.Page("ui/explore.py", title="Explore", icon=":material/tune:"),
        "model": st.Page("ui/model.py", title="Model", icon=":material/monitoring:"),
        "methodology": st.Page("ui/methodology.py", title="Methodology", icon=":material/menu_book:"),
    }
    core.PAGES.update(pages)
    core.inject_css()
    st.navigation(list(pages.values()), position="top").run()
