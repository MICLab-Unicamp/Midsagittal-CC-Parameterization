
# Colormap for visualization of parameterized images
def dark_jet_colormap(factor=0.9):
    import matplotlib.cm as cm
    import numpy as np

    # Define the original colormap
    cmap_jet = cm.get_cmap("jet")

    # Reduce brightness by scaling RGB values
    new_cmap = cmap_jet(np.linspace(0, 1, 256))  # Get colors
    new_cmap[:, :3] *= factor  # Scale RGB channels
    dark_jet = cm.colors.ListedColormap(new_cmap)

    return dark_jet


# Plot of each individual boundaries and sampling points
def vis_param(sub_path, sid, cc_msp_fname, points_sub, r_row, r_col):
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from dipy.io.image import load_nifti
    from lib_param.param_manipulation import select_points

    plt.close('all')
    fig = plt.figure(figsize=(10,10))

    cc_msp,_ = load_nifti(os.path.join(sub_path, cc_msp_fname))
    msp_slice = np.where(cc_msp==1)[0][0]
    cc_msp_reorient = np.rot90(cc_msp[msp_slice][::-1])
    cc_msp_cut,_,cut_range = crop_imgs_mask(cc_msp_reorient, cc_msp_reorient, pad=1)

    points_sub = np.array(points_sub)
    boundaries = list(points_sub[:,0])
    boundaries = np.array(boundaries+list(points_sub[:,-1][::-1]))

    idx_centerline = points_sub.shape[1]//2
    centerline = points_sub[:,idx_centerline]

    plt.subplot(2,1,1)
    plt.title(f"{sid} - boundaries (gray) and centerline (red)")
    plt.imshow(cc_msp_cut, cmap="gray")
    plt.plot(boundaries[:,0]-cut_range[1].start, boundaries[:,1]-cut_range[0].start, linestyle='solid', linewidth=1, c="gray")
    plt.plot(centerline[:,0]-cut_range[1].start, centerline[:,1]-cut_range[0].start, linestyle='solid', linewidth=1, c="r")
    plt.axis('off')

    plt.subplot(2,1,2)
    plt.title(f"{sid} - initial (gray) and selected (red) sampling points")
    plt.imshow(cc_msp_cut, cmap="gray")
    for points in points_sub:
        plt.plot(points[:,0]-cut_range[1].start, points[:,1]-cut_range[0].start, 'o', markersize=1, c="gray")
    points_sub = select_points(points_sub, r_row, r_col)
    plt.plot(points_sub[...,0]-cut_range[1].start, points_sub[...,1]-cut_range[0].start, 'o', markersize=1, color="r", linestyle='solid', linewidth=0.5)
    plt.axis('off')
    plt.tight_layout()

    return fig


# Plot of the parameterized images
def vis_param_results(title, dti_map, param_img, min_max):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from matplotlib.ticker import ScalarFormatter

    cmap = dark_jet_colormap()
    fig = plt.figure(figsize=(10,5))
    plt.title(title, fontsize=15)
    plt.imshow(param_img, vmin=min_max[0], vmax=min_max[1], cmap=cmap)
    plt.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)
    cbar = plt.colorbar(orientation='horizontal', shrink=0.4, pad=0.02)
    cbar.ax.tick_params(labelsize=15)
    cbar.locator = MaxNLocator(nbins=4)
    cbar.update_ticks()
    # Format colorbar labels as scientific notation (x × 10^y)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))  # Use scientific notation when values are <10^-2 or >10^2
    cbar.ax.xaxis.set_major_formatter(formatter)  # For horizontal colorbar
    if dti_map != "FA":
        cbar.set_label("mm²/s", fontsize=10)
    plt.tight_layout()

    return fig


# Plot of the selected parameterization configuration
def vis_param_config(np_bound, np_transv, r_row, r_col):
    import numpy as np
    import matplotlib.pyplot as plt
    from lib_param.param import param_points
    from lib_param.param_manipulation import select_points

    tmp_points,_ = param_points("", cc_msp_fname="example_cc.nii.gz", np_bound=np_bound, np_transv=np_transv)
    array_points_template = np.array(tmp_points)

    fig = plt.figure(figsize=(10,10))
    plt.title(f"({np_transv}$\\times${np_bound}) initial points $\\rightarrow$ ({np_transv-2*r_row}$\\times${np_bound-2*r_col}) selected points ")
    empty_img = np.zeros((36, 83))
    plt.imshow(empty_img, cmap="gray", vmin=-1)
    for points in array_points_template:
        plt.plot(points[:,0], points[:,1], 'o', markersize=1, c="k")
    array_points_template = select_points(array_points_template, r_row, r_col)
    plt.plot(array_points_template[...,0], array_points_template[...,1], 'o', markersize=1, color="r", linestyle='solid', linewidth=0.5)
    plt.axis('off')

    return fig


# Crop 2D images to the bounding box of the mask
def crop_imgs_mask(img, mask, pad=0):
    import numpy as np

    roi_ind = np.where(mask == 1)
    
    # Get the minimum and maximum indices in each dimension of the mask, with padding
    min_x = np.min(roi_ind[0]) - pad
    max_x = np.max(roi_ind[0]) + pad
    min_y = np.min(roi_ind[1]) - pad
    max_y = np.max(roi_ind[1]) + pad
    slice_x = slice(min_x,max_x+1)
    slice_y = slice(min_y,max_y+1)

    # Crop the image and mask using the calculated slices
    img_cut = img.copy()
    img_cut = img_cut[slice_x,slice_y]
    mask_cut = np.int8(mask[slice_x,slice_y])

    return img_cut, mask_cut, [slice_x,slice_y]