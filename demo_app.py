import streamlit as st

# Define the pages
main_page = st.Page("pages/CC_parameterization.py", title="CC Parameterization")
run_page = st.Page("pages/demo_param_setup.py", title="Parameterization Setup")

# Set up navigation
pg = st.navigation([main_page, run_page])

# Run the selected page
pg.run()