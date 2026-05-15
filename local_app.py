import streamlit as st

# Define the pages
main_page = st.Page("pages/CC_parameterization.py", title="CC Parameterization")
run_page = st.Page("pages/local_param_setup.py", title="Parameterization Setup")

# Set up navigation
pg = st.navigation([main_page, run_page])

with st.sidebar.expander(":small[Interface tips]"):
    st.markdown("""
    - :small[Refresh the page to reset the session]
    - :small[During processing, use the top-right activity indicator to monitor or stop execution]
    """)

# Run the selected page
pg.run()