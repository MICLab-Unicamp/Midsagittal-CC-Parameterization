import streamlit as st
import os
import numpy as np
from glob import glob
from dipy.io.image import load_nifti
import matplotlib.pyplot as plt
from Streamlit_local.lib_param.param import param,param_points,mean_cc,mean_param
from Streamlit_local.lib_param.util import cut_imgs_mask,vis_param,vis_param_cc
import zipfile
import tempfile

#---------------------------------------

st.title("Midsagittal Corpus Callosum Parameterization")

# ADICIONAR INFORMAÇÕES BÁSICAS SOBRE O MÉTODO!!!

#---------------------------------------

st.subheader("Method Setup")

st.write("Use the fields below to specify the inputs and output options for the method.\
         Advanced parameters can also be adjusted if needed, although the default values are recommended.")

#---------------------------------------
st.write("**INPUTS**")
#---------------------------------------

st.write("Upload the dataset as a `.zip` file. The archive should contain one folder per subject, each with the required `.nii` or `.nii.gz` files.")

uploaded_file = st.file_uploader("Upload dataset (.zip):", type="zip")

expander_fnames = st.expander("Specify required filenames", expanded=False)

cc_msp_fname = expander_fnames.text_input("Midsagittal CC mask filename:",
                                          "CC_mask_tractseg_msp.nii.gz")

tmp_dti_map_fnames = expander_fnames.text_input("DTI map filenames (comma-separated):",
                                                "DTI/dipy_dti_FA.nii.gz, DTI/dipy_dti_MD.nii.gz, DTI/dipy_dti_RD.nii.gz, DTI/dipy_dti_AD.nii.gz")
if tmp_dti_map_fnames:
    dti_map_fnames = [f.strip() for f in tmp_dti_map_fnames.split(",") if f.strip()]

tmp_dti_map_names = expander_fnames.text_input("DTI map names (comma-separated, same order as filenames):",
                                               "FA, MD, RD, AD")
if tmp_dti_map_names:
    dti_map_names = [f.strip() for f in tmp_dti_map_names.split(",") if f.strip()]

if tmp_dti_map_fnames and tmp_dti_map_names:
    if len(dti_map_fnames) != len(dti_map_names):
        st.warning("The number of DTI map filenames and names must match!")

#---------------------------------------
st.write('**OUTPUTS OPTIONS**')
#---------------------------------------

st.write("""
By default, the output is a dictionary saved as a `.npy` file in the dataset folder. It contains:
- **"points"**: coordinate points for each individual  
- **"values"**: DTI maps with parameterization results for each individual  
- **"min_max"**: minimum and maximum values for each DTI map across all individuals  
""")

expander_dict_structure = st.expander('Example of the dictionary structure', expanded=False)
expander_dict_structure.code("""
{
    "points": {
        "subj_01": [[x1, y1], [x2, y2], ...],
        "subj_02": [[x1, y1], [x2, y2], ...],
    },
    "values": {
        "FA": {
            "subj_01": np.array([...]), #shape: (Nt,Nb)
            "subj_02": np.array([...]), #shape: (Nt,Nb)
        },
        "MD": {
            "subj_01": np.array([...]), #shape: (Nt,Nb)
            "subj_02": np.array([...]), #shape: (Nt,Nb)
        }
    },
    "min_max": {
        "FA": [min_value, max_value],
        "MD": [min_value, max_value],
    }
}
""", language="python")

st.write('Other output options can be selected below.')

expander_outputs = st.expander('Output options', expanded=False)

expander_outputs.write("**Default `.npy` file(s)**")
expander_outputs.checkbox('Save parameterization data for all individuals in a single `.npy` file (default)', value=True, key="save_default")
expander_outputs.checkbox('Save parameterization data for each individual as separate `.npy` files in their respective folders', key="save_npy_sep")

#---------------------------------------
st.write('**ADVANCED PARAMETERS**')
#---------------------------------------

st.write('It is not recommended to change the parameters below, as they correspond to the proposed method by default.\
         However, it is possible to explore other parameterization settings.')

expander_adv_param = st.expander("Advanced parameters", expanded=False)
np_transv = expander_adv_param.number_input("Number of transverse points ($N_t$):", min_value=5, max_value=None, value=25)
proportion = expander_adv_param.number_input("Ratio ($r$) to define the number of boundary points ($N_b = r \cdot N_t$):", min_value=1, max_value=None, value=9)
np_bound = np_transv*proportion
# AJUSTAR CASO SEJA ZERO!!!
r_row = expander_adv_param.number_input("Number of pairs of extremity points disregarded transversely:", min_value=0, max_value=None, value=4)
r_col = expander_adv_param.number_input("Number of pairs of extremity points disregarded longitudinally:", min_value=0, max_value=None, value=12)

#---------------------------------------

st.markdown("""---""")

#---------------------------------------

st.subheader("Configuration summary")

st.write("Review the parameterization configuration, selected inputs, and output options before running the method.")

tmp_points,_ = param_points("", cc_msp_fname="Streamlit_local/example_cc.nii.gz", np_bound=np_bound, np_transv=np_transv)
array_points_template = np.array(tmp_points)
fig = plt.figure(figsize=(10,10))
plt.title(f"({np_transv}$\\times${np_bound}) initial points $\\rightarrow$ ({np_transv-2*r_row}$\\times${np_bound-2*r_col}) selected points ")
empty_img = np.zeros((36, 83))
plt.imshow(empty_img, cmap="gray", vmin=-1)
for points in array_points_template:
    plt.plot(points[:,0], points[:,1], 'o', markersize=1, c="k")
plt.plot(array_points_template[r_col:-r_col,r_row:-r_row,0], array_points_template[r_col:-r_col,r_row:-r_row,1], 'o', markersize=1, color="r", linestyle='solid', linewidth=0.5)
plt.axis('off')
st.pyplot(fig)

lines = [f'{name} ({fname})' for name, fname in zip(dti_map_names, dti_map_fnames)]
st.markdown('**DTI map(s):** ' + ', '.join(lines) + '.')

saves = ''
if st.session_state.save_default: saves+=', default `.npy`'
if st.session_state.save_npy_sep: saves+=', individual `.npy` files'

st.write('**Output(s):** ' + saves[2:] + '.')

if uploaded_file:
    st.write('**If the configuration is correct, click the button below to run the parameterization. The latest result will be displayed on the fly.**')
else:
    st.write('**:red[Upload the dataset file to allow the parcellation!]**')

st.markdown("""---""")

#---------------------------------------

if 'output' not in st.session_state:
    st.session_state.output = None

# Add a placeholder
latest_iteration = st.empty()
if st.session_state.output is not None:
    latest_iteration.write('**:green[Completed!]**')
    bar = st.progress(100)
    st.download_button(label="**Click to download the results**", data=st.session_state.output, file_name= 'output.zip', mime = 'zip', key="download_zip", type = 'primary')
else:
    bar = st.progress(0)

vis_info = st.empty()
fig_param = st.empty()

def click_button():

    # Create a temporary directory to store extracted files
    temp_dir = tempfile.TemporaryDirectory()
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        # Extract the files to the temporary directory
        zip_ref.extractall(temp_dir.name)

    temp_dir_out = tempfile.TemporaryDirectory()
    output_zip_file = os.path.join(temp_dir_out.name, 'output.zip')

    n_subs = len(glob(os.path.join(temp_dir.name, '*/')))

    # Dictionaries to store parameterization results
    dict_param_points = {}
    dict_param_maps = {}
    dict_min_max_maps = {}
    
    for i,sub_path in enumerate(sorted(glob(os.path.join(temp_dir.name, '*/')))):

        dti_map_info = (dti_map_fnames,dti_map_names)
        points_sub,dict_param_sub = param(sub_path, cc_msp_fname, dti_map_info, np_bound, np_transv)

        sub_path = os.path.normpath(sub_path)
        sid = sub_path.split(os.sep)[-1]

        dict_param_points[str(sid)] = points_sub
        for dti_map,result in dict_param_sub.items():
            if dti_map not in dict_param_maps:
                dict_param_maps[dti_map] = {}
            dict_param_maps[dti_map][str(sid)] = result

        if st.session_state.save_npy_sep:
            data_param = {"points": points_sub, "values": dict_param_sub}
            os.makedirs(os.path.join(temp_dir_out.name, f"{sid}"), exist_ok=True)
            np.save(os.path.join(temp_dir_out.name, f"{sid}", f"data_param_{sid}"), data_param)

        latest_iteration.write(f'**:blue[Computing individual {i+1} of {n_subs}, ID {sid}]**')
        bar.progress((i + 1) / n_subs)

        #---------------------------------------

        plt.close('all')
        fig = plt.figure(figsize=(10,10))

        cc_msp,_ = load_nifti(os.path.join(sub_path, cc_msp_fname))
        msp_slice = np.where(cc_msp==1)[0][0]
        cc_msp_reorient = np.expand_dims(np.rot90(cc_msp[msp_slice][::-1]), axis=0)
        cc_msp_cut,_,cut_range = cut_imgs_mask(cc_msp_reorient, cc_msp_reorient, pad=1)

        points_sub = np.array(points_sub)

        boundaries = list(points_sub[:,0])
        boundaries = np.array(boundaries+list(points_sub[:,-1][::-1]))

        idx_centerline = points_sub.shape[1]//2
        centerline = points_sub[:,idx_centerline]

        plt.subplot(2,1,1)
        plt.title(f"{sid} - boundaries (gray) and centerline (red)")
        plt.imshow(cc_msp_cut[0], cmap="gray")
        plt.plot(boundaries[:,0]-cut_range[2].start, boundaries[:,1]-cut_range[1].start, linestyle='solid', linewidth=1, c="gray")
        plt.plot(centerline[:,0]-cut_range[2].start, centerline[:,1]-cut_range[1].start, linestyle='solid', linewidth=1, c="r")
        plt.axis('off')

        plt.subplot(2,1,2)
        plt.title(f"{sid} - initial (gray) and selected (red) sampling points")
        plt.imshow(cc_msp_cut[0], cmap="gray")
        for points in points_sub:
            plt.plot(points[:,0]-cut_range[2].start, points[:,1]-cut_range[1].start, 'o', markersize=1, c="gray")
        plt.plot(points_sub[r_col:-r_col,r_row:-r_row,0]-cut_range[2].start, points_sub[r_col:-r_col,r_row:-r_row,1]-cut_range[1].start, 'o', markersize=1, color="r", linestyle='solid', linewidth=0.5)
        plt.axis('off')

        plt.tight_layout()
        vis_info.write('**Last parameterization:**')
        fig_param.pyplot(fig)

        #---------------------------------------

    # Clean up the temporary directory when the app is done
    temp_dir.cleanup()

    if st.session_state.save_default:

        for dti_map,subjects in dict_param_maps.items():
            min_val = np.inf
            max_val = -np.inf

            for arr in subjects.values():
                min_val = min(min_val, arr.min())
                max_val = max(max_val, arr.max())

            dict_min_max_maps[dti_map] = (min_val, max_val)

        data_param = {"points": dict_param_points, "values": dict_param_maps, "min_max": dict_min_max_maps}
        np.save(os.path.join(temp_dir_out.name, "data_param"), data_param)

    mean_array_points_selec,_,shape_zero_img = mean_cc(dict_param_points, r_row=r_row, r_col=r_col)
    dict_maps_mean_imgs,dict_maps_mean_cc_imgs,dict_maps_min_max = mean_param(dict_param_maps, dti_map_names, mean_array_points_selec, shape_zero_img, r_row=r_row, r_col=r_col)

    st.session_state.final_results = [dict_maps_mean_imgs,dict_maps_mean_cc_imgs,dict_maps_min_max]

    # Create a .zip file and add the files from the directory
    with zipfile.ZipFile(output_zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(temp_dir_out.name):
            for file in files:
                if not file.endswith('.zip'):
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, temp_dir_out.name))

    # Provide a download button to download the generated .zip file
    with open(output_zip_file, "rb") as f:
        zip_file_bytes = f.read()
    st.session_state.output=zip_file_bytes

    temp_dir_out.cleanup()

if "final_results" in st.session_state:
# if st.session_state.final_results is not None:

    dict_maps_mean_imgs,dict_maps_mean_cc_imgs,dict_maps_min_max = st.session_state.final_results

    st.subheader("**Average results across all individuals**")
    selected_dti_map = st.selectbox("DTI map:", dti_map_names)
    param_img = dict_maps_mean_imgs[selected_dti_map]
    param_img_cc = dict_maps_mean_cc_imgs[selected_dti_map]
    min_max = dict_maps_min_max[selected_dti_map]
    fig = vis_param(f"Average parameterized {selected_dti_map} map", param_img, min_max)
    st.pyplot(fig)
    fig = vis_param_cc(f"{selected_dti_map} values mapped to average CC", param_img_cc, min_max)
    st.pyplot(fig)

if uploaded_file:
    st.button("**Run parameterization**", type = "primary", on_click=click_button)
