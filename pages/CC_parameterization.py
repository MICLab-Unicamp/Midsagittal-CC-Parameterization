import streamlit as st

#---------------------------------------

st.title("Midsagittal Corpus Callosum Parameterization")

st.write("""
         The corpus callosum (CC) parameterization framework was proposed for the analysis of the midsagittal CC from diffusion MRI data.\
         It builds upon established ideas from morphometric CC analysis to resample the structure using a standardized grid of sampling points,\
         enabling direct point-by-point comparisons across individuals.
         
         The approach is simple, computationally efficient, avoids explicit nonlinear\
         inter-subject registration and spatial smoothing, and provides a spatially detailed characterization of diffusion properties across most\
         of the midsagittal CC.

         More information about the method is provided below.
         To run the method on your data, go to the **Parameterization Setup** page in the sidebar.
         """)

st.subheader("Overview of the method")

st.write("""
         Briefly, the approach works as follows:

         (i) Given a binary mask of the CC midsagittal section, the method generates a smoothed CC contour and separates it into\
         superior and inferior boundaries.
         """)

st.image("figs/cc_param_1.png", width=400, caption="Superior (blue) and inferior (red) boundaries defined by the genu (left)\
         and splenium (right) reference points (green).")

st.write("""
         (ii) An internal coordinate system is established by resampling each boundary at $\small N_b$ equally spaced points, connecting each pair of\
         superior-inferior points by a straight segment, and resampling each straight segment at $\small N_t$ equally spaced points.\
         To preserve the elongated geometry of the CC, $\small N_b$ is defined as a ratio of $\small N_t$ ($\small N_b = r \cdot N_t$).\
         By default, $\small r = 9$ and $\small N_t = 25$, giving $\small N_b = 225$.
         """)

st.image("figs/cc_param_2.png", width=400, caption="Sampling points over corpus callosum mask using $\small N_t = 11$ and\
         $\small N_b = 70$ for better visualization.")

st.write("""
         (iii) The parameterization points are then used to sample diffusion-derived scalar maps, yielding a standardized $\small N_t \\times N_b$\
         (default $\small 25 \\times 225$) matrix for each scalar map of each individual. This representation enables point-wise comparisons across\
         individuals, regardless of variations in CC shape and size.
         """)

st.image("figs/cc_param_3.png", width=400, caption="Sampling points over FA map using $\small N_t = 11$ and\
         $\small N_b = 70$ for better visualization.")
st.image("figs/cc_param_4.png", width=400, caption="Parameterized FA map in standardized image space using $\small N_t = 25$ and\
         $\small N_b = 225$ (default).")

st.write("""
         (iv) To mitigate partial-volume artifacts, the outermost points are disregarded, ensuring only the more robust central portion\
         of the CC is included in subsequent analyses. By default, 4 points are disregarded from each end of each matrix column and 12 points\
         are disregarded from each end of each matrix row, retaining the ($\small 17 \\times 201$) central points.
          """)

st.image("figs/cc_param_5.png", width=400, caption="Selected ($\small 17 \\times 201$) central points from the parameterized FA map.")

st.write("""
         (v) Finally, the parameterization points' coordinates are averaged across all individuals to generate a template CC representation and\
         results derived in the parameterized space can be projected back onto this template for a more intuitive visualization. Alternatively,\
         any template CC mask can be parameterized and used for the back-projection.
         """)

st.image("figs/cc_param_6.png", width=600, caption="Overview of the analysis and visualization steps using the parameterization method.")
