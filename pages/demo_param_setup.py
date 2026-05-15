import streamlit as st
import os
import numpy as np
import pandas as pd
from lib_param.param import param
from lib_param.param_manipulation import mean_cc,mean_param,map_to_voxel
from lib_param.vis import vis_param_config,vis_param,vis_param_results
from pathlib import Path,PurePosixPath
import zipfile
import tempfile

#=======================================================
st.title("Midsagittal Corpus Callosum Parameterization")
#=======================================================

st.subheader("Method Setup")

st.write("""
         Use the fields below to configure the method inputs and outputs.
         Mean parameterization results will be displayed after processing.
         """)

st.write("""
         **Required inputs:**
         - Binary midsagittal CC mask
         - Diffusion-derived maps (FA, MD, etc.)
         """)

expander_preproc = st.expander('Recommended preprocessing', expanded=False)
expander_preproc.write("""
                       - Denoising
                       - Eddy-current and motion correction
                       - Rigid alignment to the MNI152
                       - Resampling to 1.25 mm isotropic resolution
                       - Brain extraction
                       - Diffusion tensor reconstruction and diffusion maps generation
                       - Midsagittal CC segmentation ([TractSeg](https://github.com/mic-dkfz/tractseg) + post-processing was used in our experiments)

                       _The method was validated using data preprocessed with the pipeline described in the manuscript. Different preprocessing strategies may affect the results._
                       """)

st.write("")

# Validation flags
configuration_valid = True
show_config = True

with st.container(border=True):
    #==========================
    st.markdown("##### INPUTS")
    #==========================
    
    st.write("""
             Upload one or more `.zip` files containing subject folders with the required
             `.nii` or `.nii.gz` files inside them.

             Each subject folder should be named with the corresponding subject ID.
             Multiple `.zip` files can be uploaded simultaneously.
             """)

    uploaded_files = st.file_uploader("Upload `.zip` file(s):", type='zip', accept_multiple_files=True)

    # ZIP files validation
    valid_uploaded_files = []
    n_folders = 0
    if not uploaded_files:
        st.error("At least one ZIP file must be uploaded.")
        configuration_valid = False
    else:
        for uploaded_file in uploaded_files:
            try:
                with zipfile.ZipFile(uploaded_file, "r") as zf:
                    members = zf.namelist()
                    root_folders = sorted({PurePosixPath(m).parts[0] for m in members if len(PurePosixPath(m).parts) > 1})
                    n_folders += len(root_folders)
                    if len(root_folders) == 0:
                        st.warning(f"No folders were found inside _{uploaded_file.name}_")
                    else:
                        # Keep only valid ZIP files
                        valid_uploaded_files.append(uploaded_file)
            except zipfile.BadZipFile:
                st.warning(f"File _{uploaded_file.name}_ is invalid or corrupted.")
        if n_folders == 0:
            st.error("No folders were found in the ZIP file(s).")
            configuration_valid = False
        else:
            st.success(f"{n_folders} subject folder(s) detected in the ZIP file(s).")

    with st.expander("Expected `.zip` structure", expanded=False):

        st.code("""
                dataset.zip
                ├── subj_01/
                │   ├── CC_mask.nii.gz
                │   ├── FA.nii.gz
                │   ├── MD.nii.gz
                │   ├── RD.nii.gz
                │   └── AD.nii.gz
                │
                ├── subj_02/
                │   ├── CC_mask.nii.gz
                │   ├── FA.nii.gz
                │   ├── MD.nii.gz
                │   ├── RD.nii.gz
                │   └── AD.nii.gz
                """)

    #---------------------------------------

    expander_fnames = st.expander("Specify required filenames", expanded=False)

    cc_msp_fname = expander_fnames.text_input("Midsagittal CC mask filename:",
                                              "CC_mask.nii.gz")

    tmp_dti_map_fnames = expander_fnames.text_input("Diffusion map filenames (comma-separated):",
                                                    "FA.nii.gz, MD.nii.gz, RD.nii.gz, AD.nii.gz")
    if tmp_dti_map_fnames:
        dti_map_fnames = [f.strip() for f in tmp_dti_map_fnames.split(",") if f.strip()]

    tmp_dti_map_names = expander_fnames.text_input("Diffusion map names (comma-separated, same order as filenames):",
                                                   "FA, MD, RD, AD")
    if tmp_dti_map_names:
        dti_map_names = [f.strip() for f in tmp_dti_map_names.split(",") if f.strip()]

    # Diffusion maps validation
    if not tmp_dti_map_fnames.strip():
        st.error("At least one diffusion map filename must be provided.")
        configuration_valid = False
    if not tmp_dti_map_names.strip():
        st.error("At least one diffusion map name must be provided.")
        configuration_valid = False
    if tmp_dti_map_fnames and tmp_dti_map_names:
        if len(dti_map_fnames) != len(dti_map_names):
            st.error("The number of diffusion map filenames and names must match.")
            configuration_valid = False
        if len(set(dti_map_fnames)) != len(dti_map_fnames):
            st.warning("Diffusion map filenames are not unique.")
        if len(set(dti_map_names)) != len(dti_map_names):
            st.warning("Diffusion map names are not unique.")
        
    #---------------------------------------

    st.write("")
    st.write("""
            Optionally, provide a `.csv` file containing the columns `id` and `group`.
            
            - Only individuals listed in the `.csv` will be processed
            - Subject IDs must match the folder names in the dataset
            - If multiple groups are present, group-wise mean results will be displayed after processing
            """)

    csv_file = st.file_uploader("Upload `.csv` file:", type='csv')
    
    # CSV file validation
    csv_ids = None
    if csv_file:
        try:
            df_csv = pd.read_csv(csv_file, sep=None, engine='python')
            df_csv.columns = df_csv.columns.str.replace('\ufeff', '')
            required_columns = {"id", "group"}
            if not required_columns.issubset(df_csv.columns):
                st.error("CSV file must contain 'id' and 'group' columns.")
                configuration_valid = False
            else:
                # Get IDs from the input .csv file
                csv_ids = set(df_csv["id"].dropna().astype(str))
                if len(csv_ids) == 0:
                    st.error("CSV file does not contain any subject IDs.")
                    configuration_valid = False
                else:
                    st.success(f"{len(csv_ids)} subject IDs loaded from CSV.")
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            configuration_valid = False

    #---------------------------------------

    # ZIP extraction
    st.write("")
    subject_zips = {}
    if configuration_valid:
        # Create a temporary directory to store extracted files
        temp_dir = tempfile.TemporaryDirectory()
        for uploaded_file in valid_uploaded_files:
            with zipfile.ZipFile(uploaded_file, "r") as zf:
                members = zf.namelist()
                root_folders = sorted({PurePosixPath(m).parts[0] for m in members if len(PurePosixPath(m).parts) > 1})
                for sid in root_folders:
                    if csv_ids is not None and sid not in csv_ids:
                        continue
                    if sid in subject_zips:
                        st.warning(f"Subject {sid} found in multiple ZIP files: {subject_zips[sid]}, {uploaded_file.name}.\
                                   Only the first one will be processed.")
                        continue
                    subject_zips[sid] = uploaded_file.name
                    # Extract all files belonging to subject
                    subject_members = [m for m in members if PurePosixPath(m).parts[0] == sid]
                    for member in subject_members:
                        zf.extract(member, temp_dir.name)
        data_path = temp_dir.name

    # Subjects validation
    valid_subjects = []
    missing_subjects = []
    missing_csv_subjects = []
    missing_files_summary = {}
    if configuration_valid:
        data_dir = Path(data_path)
        subject_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
        dataset_subject_ids = {subj_dir.name for subj_dir in subject_dirs}
        # Check whether CSV subjects exist in dataset
        if csv_ids is not None:
            missing_csv_subjects = sorted(csv_ids - dataset_subject_ids)
        for subj_dir in subject_dirs:
            sid = subj_dir.name
            # Skip subjects not listed in CSV
            if csv_ids is not None and sid not in csv_ids:
                continue
            missing_files = []
            # CC mask
            cc_path = subj_dir / cc_msp_fname
            if not cc_path.exists():
                missing_files.append(cc_msp_fname)
            # DTI maps
            for fname in dti_map_fnames:
                map_path = subj_dir / fname
                if not map_path.exists():
                    missing_files.append(fname)
            if len(missing_files) > 0:
                missing_subjects.append(sid)
                missing_files_summary[sid] = missing_files
            else:
                valid_subjects.append(sid)
    if configuration_valid:
        if len(missing_csv_subjects) > 0:
            st.warning(f"{len(missing_csv_subjects)} subject(s) listed in the CSV were not found in the dataset and will be ignored.")
            with st.expander("Show CSV subjects not found in dataset", expanded=False):
                for sid in missing_csv_subjects:
                    st.write(f"- {sid}")
        if len(valid_subjects) == 0:
            st.error("No valid subjects were found.")
            configuration_valid = False
        else:
            st.success(f"{len(valid_subjects)} valid subjects detected.")
            if len(missing_subjects) > 0:
                st.warning(f"{len(missing_subjects)} subject(s) are missing required files and will be ignored.")
                with st.expander("Show missing files by subject", expanded=False):
                    for subj_id, files in missing_files_summary.items():
                        st.markdown(f"**{subj_id}**")
                        for f in files:
                            st.write(f"- {f}")

st.write("")

with st.container(border=True):
    #==================================
    st.markdown("##### OUTPUT OPTIONS")
    #==================================

    st.write("""
            By default, the method saves a 4D `.nii.gz` file containing the parameterization
            results of all individuals projected onto a discrete voxel grid. It also saves a `.txt`
            file listing the corresponding subject IDs.
            """)

    st.write("""
            The voxel grid resolution can be adjusted using the resolution factor below.
            For example, a factor of 2 doubles the grid dimensions, yielding smoother visualizations
            but also increasing the number of points involved in statistical analyses.
            """)

    st.write("""
            Additional output formats are available below, including individual `.nii.gz`
            files and `.npy` files containing the raw parameterization data.
            """)

    #---------------------------------------

    st.write("")
    st.write("**Default `.nii.gz` file(s)**")
    st.checkbox('Save data for all individuals in a single 4D `.nii.gz` file (default)', value=True, key="save_default")
    st.checkbox('Save data for each individual as separate `.nii.gz` files in their respective folders', key="save_nii_sep")
    mfac_param = st.number_input("Voxel grid resolution factor:", min_value=1, max_value=None, value=1,
                                            disabled=not( st.session_state.save_default or st.session_state.save_nii_sep),
                                            help="Multiplies the voxel grid dimensions by the specified factor.")
    
    st.write("")
    st.write("**Alternative `.npy` file(s)**")
    st.checkbox('Save data for all individuals in a single `.npy` file', key="save_npy")
    st.checkbox('Save data for each individual as separate `.npy` files in their respective folders', key="save_npy_sep")

    #---------------------------------------

    expander_dict_structure = st.expander('Details of the `.npy` output option', expanded=False)
    expander_dict_structure.write("""
                                This option saves the original parameterization values and sampling coordinates
                                before conversion to voxel-grid space. The output will be a dictionary saved as a `.npy`
                                file in the dataset folder. It contains:
                                
                                - **"points"**: coordinate points for each individual
                                - **"values"**: diffusion maps with parameterization results for each individual
                                - **"min_max"**: minimum and maximum values for each diffusion map across all individuals
                                
                                If the option to save individual `.npy` files is selected, the same dictionary will be saved
                                separately for each individual in their respective folders, without the min_max values.

                                An example of the dictionary structure is presented below:
                                """)
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

    selected_outputs = [st.session_state.save_default,st.session_state.save_nii_sep,st.session_state.save_npy,st.session_state.save_npy_sep]
    if not any(selected_outputs):
        st.warning("No output selected.")

st.write("")

with st.container(border=True):
    #=======================================
    st.markdown("##### ADVANCED PARAMETERS")
    #=======================================
    
    st.info("""
               The parameters below correspond to the validated configuration of the proposed method
               and should generally remain unchanged. They were selected based on experimental
               evaluation and methodological considerations.
               
               Although alternative settings can be explored, changing these parameters may affect the
               robustness and interpretability of the results. For example, reducing the number of
               discarded extremity points may include regions closer to the CC boundaries, where diffusion
               measurements tend to be less reliable. Therefore, these options are primarily intended for
               methodological exploration and experimentation.
               """)

    expander_adv_param = st.expander("Advanced parameters", expanded=False)
    np_transv = expander_adv_param.number_input("Number of transverse points ($N_t$):", min_value=5, max_value=None, value=25)
    proportion = expander_adv_param.number_input("Ratio ($r$) to define the number of boundary points ($N_b = r \cdot N_t$):", min_value=1, max_value=None, value=9)
    np_bound = np_transv*proportion
    r_row = expander_adv_param.number_input("Number of pairs of extremity points disregarded transversely:", min_value=0, max_value=None, value=4)
    r_col = expander_adv_param.number_input("Number of pairs of extremity points disregarded longitudinally:", min_value=0, max_value=None, value=12)

    if r_row * 2 >= np_transv:
        st.error("Too many transverse extremity points are being disregarded.")
        configuration_valid = False
        show_config = False
    if r_col * 2 >= np_bound:
        st.error("Too many longitudinal extremity points are being disregarded.")
        configuration_valid = False
        show_config = False

st.write("")

#====================================
st.subheader("Configuration summary")
#====================================

st.write("Review the parameterization configuration, selected inputs and outputs before running the method.")

if show_config:
    fig = vis_param_config(np_bound, np_transv, r_row, r_col)
    st.pyplot(fig)
else:
    st.error("There are configuration issues.")

lines = [f'- {name} ({fname})' for name, fname in zip(dti_map_names, dti_map_fnames)]
st.markdown('**Diffusion map(s):**\n' + '\n'.join(lines))

if csv_file:
    st.markdown('**Input `.csv` file:**\n - ' + csv_file.name)
else:
    st.markdown('**No input `.csv` file**')

saves = ''
if st.session_state.save_default: saves+=f'\n - Single `.nii.gz` file (default), resolution factor = {mfac_param}'
if st.session_state.save_nii_sep: saves+=f'\n - Individual `.nii.gz` files, resolution factor = {mfac_param}'
if st.session_state.save_npy: saves+='\n - Single `.npy` file'
if st.session_state.save_npy_sep: saves+='\n - Individual `.npy` files'

st.write('**Output(s):**\n' + saves[2:])

st.write("")

#===================================
st.subheader("Run Parameterization")
#===================================

st.info("""
         If the configuration is correct, click the button below to run the parameterization.
         Intermediate results will be displayed during processing.
         """)

st.write("")

if "clicked" not in st.session_state:
    st.session_state.clicked = False

# Run the parameterization when the button is clicked
def click_button():
    st.session_state.clicked = True

    n_subs = len(valid_subjects)

    # Create a temporary directory to store output files before zipping
    temp_dir_out = tempfile.TemporaryDirectory()
    output_zip_file = os.path.join(temp_dir_out.name, 'output.zip')

    # Dictionaries to store parameterization results
    dict_param_points = {}
    dict_param_maps = {}
    dict_min_max_maps = {}
    
    # Run the parameterization for each individual
    for i,sid in enumerate(valid_subjects):
        sub_path = os.path.join(data_path, sid)

        # Parameterization
        dti_map_info = (dti_map_fnames, dti_map_names)
        points_sub,dict_param_sub = param(sub_path, cc_msp_fname, dti_map_info, np_bound, np_transv)

        # Store results in dictionaries
        dict_param_points[str(sid)] = points_sub
        for dti_map,result in dict_param_sub.items():
            if dti_map not in dict_param_maps:
                dict_param_maps[dti_map] = {}
            dict_param_maps[dti_map][str(sid)] = result

        # Save individual parameterization results in separate .npy files
        if st.session_state.save_npy_sep:
            data_param = {"points": points_sub, "values": dict_param_sub}
            sub_path_param = os.path.join(temp_dir_out.name, sid, "param")
            os.makedirs(sub_path_param, exist_ok=True)
            np.save(os.path.join(sub_path_param, f"data_param_{sid}"), data_param)

        # Update progress bar
        latest_iteration.write(f'**:blue[Computing individual {i+1} of {n_subs}, ID {sid}]**')
        bar.progress((i + 1) / n_subs)

        # Plot each individual boundaries and sampling points
        fig = vis_param(sub_path, sid, cc_msp_fname, points_sub, r_row, r_col)
        vis_info.write('**Last parameterization:**')
        fig_param.pyplot(fig)

    # Save parameterization results in a single .npy file
    if st.session_state.save_npy:

        # Compute minimum and maximum values for each DTI map across all individuals
        for dti_map,subjects in dict_param_maps.items():
            min_val = np.inf
            max_val = -np.inf
            for arr in subjects.values():
                min_val = min(min_val, arr.min())
                max_val = max(max_val, arr.max())
            dict_min_max_maps[dti_map] = (min_val, max_val)

        # Save the parameterization results and min-max values in a single .npy file
        data_param = {"points": dict_param_points, "values": dict_param_maps, "min_max": dict_min_max_maps}
        np.save(os.path.join(temp_dir_out.name, "data_param"), data_param)

    if st.session_state.save_default or st.session_state.save_nii_sep:
        # Compute the mean of the coordinate points to obtain the "mean CC" and define the shape for plotting
        mean_array_points_selec,shape_zero_img = mean_cc(dict_param_points, r_row, r_col, mfac_param)

        # Save parameterization results in a single .nii.gz file (default)
        if st.session_state.save_default:
            map_to_voxel(dict_param_maps, mean_array_points_selec, shape_zero_img, n_subs, r_row, r_col, temp_dir_out.name, mfac_param)
            # Save ID list in a .txt file for reference
            with open(os.path.join(temp_dir_out.name, "ids.txt"), "w") as f:
                for sid in valid_subjects:
                    f.write(f"{sid}\n")

        # Save individual parameterization results in separate .nii.gz files
        if st.session_state.save_nii_sep:
            for sid in dict_param_points:
                sub_path_param = os.path.join(temp_dir_out.name, sid, "param")
                os.makedirs(sub_path_param, exist_ok=True)
                dict_param_sub = {dti_map: {sid: dict_param_maps[dti_map][sid]} for dti_map in dict_param_maps}
                map_to_voxel(dict_param_sub, mean_array_points_selec, shape_zero_img, 1, r_row, r_col, sub_path_param, mfac_param)

    # Store results in session state to be displayed at the end of the computation
    st.session_state.final_results = [dict_param_points,dict_param_maps]

    # Create a .zip file and add the files from the directory
    with zipfile.ZipFile(output_zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(temp_dir_out.name):
            for file in files:
                if not file.endswith('.zip'):
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, temp_dir_out.name))

    # Provide a download button to download the generated .zip file
    if any(selected_outputs):
        with open(output_zip_file, "rb") as f:
            zip_file_bytes = f.read()
        st.session_state.output=zip_file_bytes

    temp_dir_out.cleanup()

if not configuration_valid:
    st.error("Please resolve the issues above before running the method.")

# Button to run the parameterization
st.button("**Run parameterization**", type="primary", on_click=click_button, disabled=not configuration_valid)

# Add placeholders
latest_iteration = st.empty()
bar = st.empty()
vis_info = st.empty()
fig_param = st.empty()

# Initialize and finalize progress bar
if st.session_state.clicked:
    latest_iteration.write('**:green[Completed!]**')
    bar.progress(100)
    if "output" in st.session_state:
        st.write("")
        st.download_button(label="**Click to download the results**", data=st.session_state.output, file_name='output.zip', mime='zip', key="download_zip", type='primary')
else:
    bar.progress(0)

# Compute and show mean results after running the parameterization
if "final_results" in st.session_state:
    st.write("")
    st.subheader("**Average parameterization results**")

    dict_param_points,dict_param_maps = st.session_state.final_results

    selected_dti_map = st.selectbox("Diffusion map:", dti_map_names)
    mfac_vis = st.number_input("Voxel grid resolution factor (only for visualization):", min_value=1, max_value=None, value=mfac_param)

    # Compute the mean of the coordinate points to obtain the "mean CC" and define the shape for plotting
    mean_array_points_selec,shape_zero_img = mean_cc(dict_param_points, r_row, r_col, mfac_vis)

    # Check whether to show mean results by group or across all individuals
    use_groups = False
    if csv_file:
        group_list = df_csv["group"].unique().tolist()
        if len(group_list) > 1:
            use_groups = True

    # Store computed results for each group
    group_results = {}
    min_max = None

    if use_groups:
        st.markdown("##### Average results by group")

        for group_name in group_list:
            # IDs belonging to the group
            group_ids = df_csv.loc[df_csv["group"] == group_name, "id"].tolist()

            # Filter dict_param_maps
            dict_param_maps_group = {
                dti_map: {
                    sid: dict_param_maps[dti_map][sid]
                    for sid in dict_param_maps[dti_map]
                    if sid in group_ids
                }
                for dti_map in dict_param_maps
            }

            # Compute the mean of the parameterization results across individuals of the group
            dict_maps_mean_imgs,dict_maps_mean_cc_imgs,dict_maps_min_max = mean_param(dict_param_maps_group, dti_map_names, mean_array_points_selec, shape_zero_img, r_row, r_col)

            # Store results
            group_results[group_name] = {
                "mean_imgs": dict_maps_mean_imgs,
                "mean_cc_imgs": dict_maps_mean_cc_imgs,
            }
            if min_max is None:
                min_max = dict_maps_min_max[selected_dti_map]
            else:
                tmp_min_max = dict_maps_min_max[selected_dti_map]
                min_max = [np.min([min_max[0], tmp_min_max[0]]), np.max([min_max[1], tmp_min_max[1]])]

    else:
        st.markdown("##### Average results across all individuals")

        # Compute the mean of the parameterization results across individuals
        dict_maps_mean_imgs,dict_maps_mean_cc_imgs,dict_maps_min_max = mean_param(dict_param_maps, dti_map_names, mean_array_points_selec, shape_zero_img, r_row, r_col)

        # Store results
        group_results["all"] = {
            "mean_imgs": dict_maps_mean_imgs,
            "mean_cc_imgs": dict_maps_mean_cc_imgs,
        }
        min_max = dict_maps_min_max[selected_dti_map]

    # Show all average parameterized images first
    for group_name in group_results:
        title = f"Average parameterized {selected_dti_map} map"
        if use_groups:
            title += f" ({group_name})"
        param_img = group_results[group_name]["mean_imgs"][selected_dti_map]
        fig = vis_param_results(title, selected_dti_map, param_img, min_max)
        st.pyplot(fig)

    # Then show all average CC images
    for group_name in group_results:
        title = f"{selected_dti_map} mapped to average CC"
        if use_groups:
            title += f" ({group_name})"
        param_img_cc = group_results[group_name]["mean_cc_imgs"][selected_dti_map]
        fig = vis_param_results(title, selected_dti_map, param_img_cc, min_max)
        st.pyplot(fig)
