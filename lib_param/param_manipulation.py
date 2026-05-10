
# Compute the mean of the coordinate points to obtain the "mean CC"
def mean_cc(dict_param_points, r_row, r_col, m_fac):
    import numpy as np
    from math import ceil

    # Convert dict of points to array
    array_points = np.stack(list(dict_param_points.values()))

    # Mean to obtain the "mean CC"
    mean_array_points = array_points.mean(axis=0)

    # Reduce the coordinate values for easier visualization and manipulation
    mean_array_points = reduce_coord_values(mean_array_points)

    # Multiplying factor for creating the mapped CC image
    mean_array_points = mean_array_points*m_fac

    # Define the shape used for plotting the points
    shape_zero_img = (ceil(mean_array_points[...,1].max())+np.max([m_fac//2,2]),ceil(mean_array_points[...,0].max())+np.max([m_fac//2,2]))

    # Disregard the selected number of rows and columns
    mean_array_points_selec = select_points(mean_array_points, r_row, r_col)

    return mean_array_points_selec, shape_zero_img


# Reduce the coordinate values for easier visualization and manipulation
def reduce_coord_values(mean_array_points):
    from math import floor

    reduc_0 = floor(mean_array_points[...,0].min()) -1
    reduc_1 = floor(mean_array_points[...,1].min()) -1
    mean_array_points[...,0] = mean_array_points[...,0] - reduc_0
    mean_array_points[...,1] = mean_array_points[...,1] - reduc_1

    return mean_array_points


# Disregard the selected number of rows and columns
def select_points(array_points, r_row, r_col):

    if r_row != 0 and r_col != 0:
        array_points_selec = array_points[r_col:-r_col,r_row:-r_row]
    elif r_row != 0:
        array_points_selec = array_points[:,r_row:-r_row]
    elif r_col != 0:
        array_points_selec = array_points[r_col:-r_col]
    else:
        array_points_selec = array_points

    return array_points_selec


# Map parameterization results back to the image space
def map_to_cc(array_points_selec, mean_array_param, shape_zero_img):
    import numpy as np
    import numpy.ma as ma
    import scipy.interpolate as interp
    import cv2 as cv
    
    zero_img = np.zeros(shape_zero_img)
    masked_img = ma.masked_array(zero_img, zero_img==0).copy()
    masked_count = ma.masked_array(zero_img, zero_img==0).copy()
    n_points = len(array_points_selec[0])

    # Store valid (x, y) coordinates and their values
    points = []
    values = []

    # Map parameterization values back to the image space
    for i in range(len(array_points_selec)):
        for j in range(n_points):
            # Get the pixel coordinates
            x, y = array_points_selec[i][j]
            # Round to the nearest integer
            x = int(round(x))
            y = int(round(y))

            # Ensure the coordinates are within bounds
            if 0 <= x < masked_img.shape[1] and 0 <= y < masked_img.shape[0]:
                # Assign the parameterization value
                if masked_img.mask[y, x] == True:
                    masked_img[y, x] = mean_array_param[j, i]
                    masked_count[y, x] = 1
                else:
                    masked_img[y, x] += mean_array_param[j, i]
                    masked_count[y, x] += 1
                if [x, y] not in points:
                    points.append([x, y])

    # Mean of values if more than one point in a voxel
    masked_img = masked_img/masked_count

    for p in points:
        values.append(masked_img[p[1], p[0]])

    # Convert to numpy arrays
    points = np.array(points)
    values = np.array(values)

    # Generate grid points
    grid_x, grid_y = np.meshgrid(np.arange(masked_img.shape[1]), np.arange(masked_img.shape[0]))

    # Interpolate missing values
    interpolated_img = interp.griddata(points, values, (grid_x, grid_y), method='linear')

    # Select boundary points
    mean_array_points_bound_1 = array_points_selec[:,0]
    mean_array_points_bound_2 = array_points_selec[:,-1]
    mean_array_points_bound_3 = array_points_selec[0]
    mean_array_points_bound_4 = array_points_selec[-1]

    # Concatenate boundary points in the correct order to form a closed contour
    mean_array_points_bound_1_rev = mean_array_points_bound_1[::-1]
    list_boundary = list(mean_array_points_bound_1_rev)
    list_boundary = list_boundary + list(mean_array_points_bound_3)
    list_boundary = list_boundary + list(mean_array_points_bound_2)
    mean_array_points_bound_4_rev = mean_array_points_bound_4[::-1]
    list_boundary = list_boundary + list(mean_array_points_bound_4_rev)
    list_boundary = np.array(list_boundary)

    # Create a CC mask using the boundary points
    mask = np.zeros(shape_zero_img, dtype=np.uint8)
    param_contour = np.round(np.array(list_boundary))
    cv.fillPoly(mask, [param_contour.astype(np.int32)], color=1)

    # Apply the CC mask to the interpolated image
    masked_interpolated_img = ma.masked_array(interpolated_img, mask==0)

    return mask,masked_interpolated_img


# Compute the mean of the parameterization results across individuals
def mean_param(dict_param_maps, dti_map_names, array_points_selec, shape_zero_img, r_row, r_col):
    import numpy as np

    dict_maps_mean_imgs = {}
    dict_maps_mean_cc_imgs = {}
    dict_maps_min_max = {}

    for dti_map in dti_map_names:

        dict_param = dict_param_maps[dti_map]

        tmp_min = np.inf
        tmp_max = -np.inf

        # Convert dict of parameterization results to array and compute the mean across individuals
        array_param = np.stack(list(dict_param.values()))
        mean_array_param = array_param.mean(axis=0)
        # Disregard the selected number of rows and columns
        mean_array_param = select_points(mean_array_param, r_col, r_row)

        # Update global min and max values across individuals
        if mean_array_param.min() < tmp_min:
            tmp_min = mean_array_param.min()
        if mean_array_param.max() > tmp_max:
            tmp_max = mean_array_param.max()

        # Map mean parameterization results back to the image space
        _,masked_interpolated_img = map_to_cc(array_points_selec, mean_array_param, shape_zero_img)

        # Store mean parameterization results and global min and max values in dictionaries
        dict_maps_min_max[dti_map] = [tmp_min, tmp_max]
        dict_maps_mean_imgs[dti_map] = mean_array_param
        dict_maps_mean_cc_imgs[dti_map] = masked_interpolated_img

    return dict_maps_mean_imgs, dict_maps_mean_cc_imgs, dict_maps_min_max


def map_to_voxel(dict_param_maps, mean_array_points_selec, shape_zero_img, n_subs, r_row, r_col, base_path, mfac_param):
    import numpy as np
    import os
    from dipy.io.image import save_nifti

    tmp_dict = dict_param_maps[next(iter(dict_param_maps))]
    base_dtype = tmp_dict[next(iter(tmp_dict))].dtype
    affine = np.eye(4)
    new_shape_zero_img = (shape_zero_img[1],shape_zero_img[0])

    for dti_map in dict_param_maps:
        
        if n_subs == 1:
            all_param_map = np.zeros(np.append(3,new_shape_zero_img), dtype=base_dtype)
        else:
            all_param_map = np.zeros(np.append(np.append(3,new_shape_zero_img),n_subs), dtype=base_dtype)
        dict_param = dict_param_maps[dti_map]

        for i,sid in enumerate(dict_param):
            tmp_param = dict_param[sid]
            tmp_param = select_points(tmp_param, r_col, r_row)

            mask,masked_interpolated_img = map_to_cc(mean_array_points_selec, tmp_param, shape_zero_img)

            masked_interpolated_img[masked_interpolated_img.mask == True] = 0
            if n_subs == 1:
                all_param_map[1] = np.rot90(masked_interpolated_img[::-1])
            else:
                all_param_map[1,...,i] = np.rot90(masked_interpolated_img[::-1])

        if n_subs == 1:
            save_nifti(os.path.join(base_path, f"{dti_map}_{mfac_param}.nii.gz"), all_param_map, affine)
        else:
            save_nifti(os.path.join(base_path, f"all_{dti_map}_{mfac_param}.nii.gz"), all_param_map, affine)

    tmp_mask = np.zeros(np.append(3,new_shape_zero_img), dtype=np.uint8)
    tmp_mask[1] = np.rot90(mask[::-1])
    save_nifti(os.path.join(base_path, f"param_mask_{mfac_param}.nii.gz"), tmp_mask, affine)

    return
