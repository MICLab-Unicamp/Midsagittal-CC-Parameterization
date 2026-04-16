
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

# Plot of the parameterized images
def vis_param(title, param_img, min_max):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    cmap = dark_jet_colormap()

    fig = plt.figure(figsize=(10,3))
    plt.title(title, fontsize=15)
    plt.imshow(param_img, vmin=min_max[0], vmax=min_max[1], cmap=cmap)
    plt.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)
    cbar = plt.colorbar(orientation='horizontal', shrink=0.5, pad=0.02)
    cbar.ax.tick_params(labelsize=15)
    cbar.locator = MaxNLocator(nbins=4)
    cbar.update_ticks()

    plt.tight_layout()

    return fig


# Plot of the parameterized values mapped to mean CC
# Boundaries using only the points taken into consideration
def vis_param_cc(title, param_img_cc, min_max):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import matplotlib.patches as patches

    cmap = dark_jet_colormap()

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.title(title, fontsize=15)
    plt.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)

    # patch_cut = patches.Polygon(param_img_cc[1], facecolor='none')
    # ax.add_patch(patch_cut)
    plt.imshow(param_img_cc[-1], vmin=min_max[0], vmax=min_max[1], cmap=cmap)#, clip_path=patch_cut, clip_on=True)

    # plt.plot(param_img_cc[1][:,0], param_img_cc[1][:,1], linewidth=3)

    cbar = plt.colorbar(orientation='horizontal', shrink=0.5, pad=0.02)
    cbar.ax.tick_params(labelsize=15)
    cbar.locator = MaxNLocator(nbins=4)
    cbar.update_ticks()
    
    plt.tight_layout()

    return fig


def cut_imgs_mask(img, mask, pad=0, bin=True):
    import numpy as np

    roi_ind = np.where(mask == 1)
    
    # Pega os máximos e mínimos em cada dimensão (extremidades da máscara)
    min_x = np.min(roi_ind[0]) - pad
    max_x = np.max(roi_ind[0]) + pad
    min_y = np.min(roi_ind[1]) - pad
    max_y = np.max(roi_ind[1]) + pad
    min_z = np.min(roi_ind[2]) - pad
    max_z = np.max(roi_ind[2]) + pad
    slice_x = slice(min_x,max_x+1)
    slice_y = slice(min_y,max_y+1)
    slice_z = slice(min_z,max_z+1)

    # Bounding box em torno da máscara para evitar cálculos desnecessários
    img_cut = img.copy()
    img_cut = img_cut[slice_x,slice_y,slice_z]
    mask_cut = np.int8(mask[slice_x,slice_y,slice_z])

    return img_cut, mask_cut, [slice_x,slice_y,slice_z]