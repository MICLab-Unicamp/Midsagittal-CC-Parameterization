
# CC boundaries definition
def CC_boundaries(cc_msp_reorient, npoints=225, s_factor=1):
    import numpy as np
    import cv2 as cv
    from scipy import interpolate
    import math

    #--------------------
    # Initial CC contour
    #--------------------

    # Changing the dtype of the CC mask
    cc_msp_reorient = np.array(cc_msp_reorient, dtype='uint8')

    # Finding the contour voxels of the CC mask
    contours,_ = cv.findContours(cc_msp_reorient, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if len(contours) != 1:
        print("Warning! The number of contours found is not 1. The contour with the largest area will be selected.")
    contour = max(contours, key=cv.contourArea)
    boundary = contour[:,0][:,::-1]

    #---------------------
    # Smoothed CC contour
    #---------------------

    # Extending the boundary to obtain a smooth curve in all points
    ext_points = boundary.shape[0]//5
    boundary_ext = np.zeros((boundary.shape[0]+ext_points, boundary.shape[1]))
    boundary_ext[:boundary.shape[0]] = boundary
    boundary_ext[boundary.shape[0]:] = boundary[0:ext_points]

    # Defining the number of points and the amount of smoothness of the spline
    npoints_boundary = boundary_ext.shape[0]*5
    s = (npoints_boundary/50)*s_factor

    # Obtaining a spline from the extended boundary points
    spline,_ = interpolate.make_splprep(boundary_ext.transpose(),s=s)
    # Evaluating the spline using a large number of points
    unew = np.linspace(0,1,npoints_boundary)
    yInter,xInter = spline(unew)

    # Identifying indexes to remove extra points
    min_dist = np.inf
    # Number of extra points to check (last points of the curve, corresponding to the ext_points)
    points_to_test = int(np.round((npoints_boundary*ext_points)/boundary_ext.shape[0]))
    # Number of neighbors to consider in each side of the point to be tested
    n_points_neighbors = 5
    # Test all extra points (last points of the curve), except for the ones in the extremities (n_points_neighbors in each side)
    for i in range(points_to_test-(n_points_neighbors*2)):
        idx_test_last = -(i+1+n_points_neighbors)

        # Considering the same number of points (points_to_test, except for the ones in the extremities), but for the first points of the curve
        for idx_test_first in range(n_points_neighbors,points_to_test-n_points_neighbors):
            
            # Computes the distance between each last point (and its neighbors) and each first point (and its neighbors)
            sum_dist = 0
            for k in range(-n_points_neighbors,n_points_neighbors+1):
                tmp_first = [xInter[idx_test_first+k],yInter[idx_test_first+k]]
                tmp_last = [xInter[idx_test_last+k],yInter[idx_test_last+k]]
                tmp_dist = math.dist(tmp_first, tmp_last)
                sum_dist += tmp_dist

            # If the accumulated distance is the smallest one so far, replace the min_dist and keep the index values
            if sum_dist < min_dist:
                min_dist = sum_dist
                idxs = (idx_test_first,idx_test_last)

    # Removing extra points
    new_xInter = xInter[idxs[0]:idxs[1]+1]
    new_yInter = yInter[idxs[0]:idxs[1]+1]

    #--------------------------------
    # Curvature along the CC contour
    #--------------------------------

    # Computing first and second derivatives
    d_spline = spline.derivative(1)
    d2_spline = spline.derivative(2)
    dy, dx = d_spline(unew)  # First derivative
    d2y, d2x = d2_spline(unew)  # Second derivative

    # Computing the curvature
    curvature = -(dx * d2y - dy * d2x) / (dx**2 + dy**2) ** (3 / 2)

    #-----------------------------------------------------
    # Limiting the points at which to check the curvature
    #-----------------------------------------------------

    # Get indices of the anterior half
    anterior_indices = np.where(new_xInter < np.mean(new_xInter)) 
    # Add the first index to the end to close the "loop" for distance calculation
    anterior_indices_ext = np.append(anterior_indices[0], anterior_indices[0][0])
    # Get the distance between points and find the largest gap
    ant_points = np.column_stack((new_xInter[anterior_indices_ext], new_yInter[anterior_indices_ext]))
    ant_points_dist = np.linalg.norm(np.diff(ant_points, axis=0), axis=1)
    max_dist_idx = np.argmax(ant_points_dist)
    # Identify the two indices involved in the gap
    idx_a = anterior_indices_ext[max_dist_idx]
    idx_b = anterior_indices_ext[max_dist_idx + 1]
    # Compare Y coordinates to find the "lower" point (higher Y)
    if new_yInter[idx_a] > new_yInter[idx_b]:
        max_x_low_anterior_index = idx_a
    else:
        max_x_low_anterior_index = idx_b

    # Find the max Y within the anterior subset
    y_sub = new_yInter[anterior_indices[0]]
    max_y_val = np.max(y_sub)
    # Get the original index
    max_y_anterior_index = anterior_indices[0][np.where(y_sub == max_y_val)[0][0]]

    #-------------------------------
    # Finding the anterior endpoint
    #-------------------------------

    if max_y_anterior_index > max_x_low_anterior_index:
        print("Warning! Problem with the definition of the anterior endpoint!")

    max_curv_idx = np.argmax(curvature[idxs[0]:idxs[1]][max_y_anterior_index:max_x_low_anterior_index]) + (max_y_anterior_index)

    # Getting the anterior endpoint of the CC (point of maximum curvature)
    anterior_endpoint = (new_xInter[max_curv_idx], new_yInter[max_curv_idx])

    #--------------------------------
    # Finding the posterior endpoint
    #--------------------------------

    # Identifying the posterior half of the boundary
    posterior_indices = np.where(new_xInter > np.mean(new_xInter))
    # Identifying the index of the maximum y-coordinate considering the posterior half of the boundary
    post_end_idx = posterior_indices[0][np.argmax(new_yInter[posterior_indices])]
    # Getting the posterior endpoint of the CC (lowest point of the splenium)
    posterior_endpoint = (new_xInter[post_end_idx], new_yInter[post_end_idx])

    #-----------------------------------------
    # Defining the upper and lower boundaries
    #-----------------------------------------

    # Getting the points of the lower and upper boundaries using the anterior and posterior endpoints
    if max_curv_idx<post_end_idx:
        lower_bound_x = new_xInter[max_curv_idx:post_end_idx+1]
        lower_bound_y = new_yInter[max_curv_idx:post_end_idx+1]
        upper_bound_x = np.concatenate((new_xInter[:max_curv_idx+1][::-1],new_xInter[post_end_idx:-1][::-1]))
        upper_bound_y = np.concatenate((new_yInter[:max_curv_idx+1][::-1],new_yInter[post_end_idx:-1][::-1]))
    else:
        lower_bound_x = np.concatenate((new_xInter[max_curv_idx:-1],new_xInter[:post_end_idx+1]))
        lower_bound_y = np.concatenate((new_yInter[max_curv_idx:-1],new_yInter[:post_end_idx+1]))
        upper_bound_x = new_xInter[post_end_idx:max_curv_idx+1][::-1]
        upper_bound_y = new_yInter[post_end_idx:max_curv_idx+1][::-1]

    # Obtaining splines from the boundary points and evaluating them using the desired number of points for the parameterization
    # To define the final upper and lower boundary points
    unew = np.linspace(0,1,npoints)
    spline,u = interpolate.make_splprep(np.array([lower_bound_y,lower_bound_x]),s=0)
    yInter_low,xInter_low = spline(unew)
    spline,u = interpolate.make_splprep(np.array([upper_bound_y,upper_bound_x]),s=0)
    yInter_up,xInter_up = spline(unew)

    # Getting the centerline points from the upper and lower points
    xInter_center = (xInter_low+xInter_up)/2
    yInter_center = (yInter_low+yInter_up)/2

    return xInter_low, yInter_low, xInter_up, yInter_up, xInter_center, yInter_center, new_xInter, new_yInter


# Parameterization points definition
def param_points(sub_path, cc_msp_fname="CC_mask_tractseg_msp.nii.gz", np_bound=225, np_transv=25, s_fac=1):
    import numpy as np
    import os
    from dipy.io.image import load_nifti

    # Loading CC mask (only MSP)
    cc_msp,_ = load_nifti(os.path.join(sub_path, cc_msp_fname))

    # Selecting only the midsagittal "plane" and reorienting the 2D image to obtain the CC boundaries
    msp_slice = np.where(cc_msp==1)[0][0]
    cc_msp_reorient = np.rot90(cc_msp[msp_slice][::-1])

    # Obtaining the CC boundaries
    xInter_low,yInter_low,xInter_up,yInter_up,_,_,_,_ = CC_boundaries(cc_msp_reorient.copy(), np_bound, s_fac)

    tmp_points_sub = []

    # For each pair of points (upper and lower boundaries)
    for i in range(len(xInter_up)):
        tmp_points = []
        # Computes the coordinates of the intermediate points in the tranverse lines
        for j in range(np_transv):
            # Check if the points of the upper and lower boundaries are equal
            if (xInter_up[i] == xInter_low[i]) and (yInter_up[i] == yInter_low[i]):
                x = xInter_up[i]
                y = yInter_up[i]
            # Otherwise computes the intermediate points
            else:
                x = (xInter_up[i]) + j*((xInter_low[i]-xInter_up[i])/(np_transv-1))
                m = (yInter_up[i]-yInter_low[i])/(xInter_up[i]-xInter_low[i])
                y = m*(x - xInter_up[i]) + yInter_up[i]

            tmp_points.append([x,y])

        tmp_points_sub.append(tmp_points)

    return tmp_points_sub, msp_slice


# CC parameterization
def param(sub_path, cc_msp_fname="CC_mask_tractseg_msp.nii.gz", dti_map_info=(["DTI/dipy_dti_FA.nii.gz"],["FA"]),
          np_bound=225, np_transv=25, s_fac=1):
    import numpy as np
    import os
    from dipy.io.image import load_nifti
    from scipy.ndimage import map_coordinates
    
    dti_map_fnames,dti_map_names = dti_map_info

    # Obtaining the parameterization points
    tmp_points_sub, msp_slice = param_points(sub_path, cc_msp_fname, np_bound, np_transv, s_fac)
    tmp_dict_param = {}

    for j,dti_map in enumerate(dti_map_fnames):
        map_name = dti_map_names[j]

        # Loading DTI map
        img,_ = load_nifti(os.path.join(sub_path, dti_map))

        # Selecting only the midsagittal slice and reorienting the 2D image
        img_msp_reorient = np.rot90(img[msp_slice][::-1])

        # Calculating the parameterized map
        img_param = np.zeros((np_transv, np_bound))
        
        # Obtaining DTI map values at each parameterization point
        for k in range(len(tmp_points_sub)):
            values = map_coordinates(img_msp_reorient, [np.array(tmp_points_sub[k])[:,1], np.array(tmp_points_sub[k])[:,0]], order=1)
            img_param[:,k] = values

        tmp_dict_param[map_name] = img_param

    return tmp_points_sub, tmp_dict_param


def mean_cc(dict_param_points, m_fac=10, r_row=1, r_col=2, ext_vis=0):
    import numpy as np

    # Convert dict of points to array
    array_points = np.stack(list(dict_param_points.values()))

    # Mean to obtain the "mean CC"
    mean_array_points = array_points.mean(axis=0)

    # Multiplying factor for creating the mapped CC image
    mean_array_points = mean_array_points*m_fac

    # Reducing the coordinates values for easier visualization and manipulation
    mean_array_points,_,_ = reduce_coord_values(mean_array_points, m_fac)

    # Defining the shape used for plotting the points
    shape_zero_img = (int(mean_array_points[...,1].max())+m_fac//2+ext_vis,int(mean_array_points[...,0].max())+m_fac//2+ext_vis)

    mean_array_points_selec, _, array_bound_removed = select_points(mean_array_points, r_row, r_col)

    return mean_array_points_selec, array_bound_removed, shape_zero_img


def reduce_coord_values(mean_array_points, m_fac):

    # Reducing the coordinates values for easier visualization and manipulation
    reduc_0 = int(mean_array_points[...,0].min())-m_fac//2
    reduc_1 = int(mean_array_points[...,1].min())-m_fac//2
    mean_array_points[...,0] = mean_array_points[...,0] - reduc_0
    mean_array_points[...,1] = mean_array_points[...,1] - reduc_1

    return mean_array_points, reduc_0, reduc_1


def select_points(array_points, r_row=1, r_col=2):
    import numpy as np

    # Disregarding the first and last rows and the two first and two last columns
    if r_row != 0 and r_col != 0:
        array_points_selec = array_points[r_col:-r_col,r_row:-r_row]
    elif r_row != 0:
        array_points_selec = array_points[:,r_row:-r_row]
    elif r_col != 0:
        array_points_selec = array_points[r_col:-r_col]
    else:
        array_points_selec = array_points

    # Getting the removed points for visualization
    if r_row != 0:
        array_points_remov_1 = array_points[:,0:r_row]
        array_points_remov_2 = array_points[:,-r_row:]
    else:
        array_points_remov_1 = []
        array_points_remov_2 = []

    if r_col != 0:
        array_points_remov_3 = array_points[0:r_col]
        array_points_remov_4 = array_points[-r_col:]
    else:
        array_points_remov_3 = []
        array_points_remov_4 = []

    points_removed = (array_points_remov_1, array_points_remov_2, array_points_remov_3, array_points_remov_4)

    # Putting the removed points together in a way to visualize the borders
    if r_col != 0:
        if r_col == 1:
            array_points_remov_1_rev = array_points_remov_1[(r_col-1):,(r_row-1)][::-1]
        else:
            array_points_remov_1_rev = array_points_remov_1[(r_col-1):-(r_col-1),(r_row-1)][::-1]
        list_removed = list(array_points_remov_1_rev)
    else:
        list_removed = []

    if r_row != 0:
        if r_row == 1:
            list_removed = list_removed + list(array_points_remov_3[(r_col-1),(r_row-1):])
        else:
            list_removed = list_removed + list(array_points_remov_3[(r_col-1),(r_row-1):-(r_row-1)])

    if r_col != 0:
        if r_col == 1:
            list_removed = list_removed + list(array_points_remov_2[(r_col-1):,-r_row])
        else:
            list_removed = list_removed + list(array_points_remov_2[(r_col-1):-(r_col-1),-r_row])

    if r_row != 0:
        if r_row == 1:
            array_points_remov_4_rev = array_points_remov_4[-(r_col),(r_row-1):][::-1]
        else:
            array_points_remov_4_rev = array_points_remov_4[-(r_col),(r_row-1):-(r_row-1)][::-1]
        list_removed = list_removed + list(array_points_remov_4_rev)

    # Convert list of removed points to array
    array_bound_removed = np.array(list_removed)

    return array_points_selec, points_removed, array_bound_removed


def map_to_cc(array_points_selec, mean_array_param, shape_zero_img, array_bound_removed=None):
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

    # Map img_param values back to the image
    for i in range(len(array_points_selec)):
        for j in range(n_points):
            # Get the pixel coordinates
            x, y = array_points_selec[i][j]
            x = int(round(x))  # Round to the nearest integer
            y = int(round(y))

            # Ensure the coordinates are within bounds
            if 0 <= x < masked_img.shape[1] and 0 <= y < masked_img.shape[0]:
                # Assign the value from img_param
                if masked_img.mask[y, x] == True:
                    masked_img[y, x] = mean_array_param[j, i]
                    masked_count[y, x] = 1
                else:
                    masked_img[y, x] += mean_array_param[j, i]
                    masked_count[y, x] += 1
                if [x, y] not in points:
                    points.append([x, y])
                # values.append(mean_array_param[j, i])

    # Mean of values if more than one point in a voxel
    masked_img = masked_img/masked_count

    for p in points:
        values.append(masked_img[p[1], p[0]])

    # Convert to numpy arrays
    points = np.array(points)
    values = np.array(values)

    # Generate grid points for the full image
    grid_x, grid_y = np.meshgrid(np.arange(masked_img.shape[1]), np.arange(masked_img.shape[0]))

    # Interpolate missing values
    interpolated_img = interp.griddata(points, values, (grid_x, grid_y), method='linear')

    if array_bound_removed is None:
        mean_array_points_bound_1 = array_points_selec[:,0]
        mean_array_points_bound_2 = array_points_selec[:,-1]
        mean_array_points_bound_3 = array_points_selec[0]
        mean_array_points_bound_4 = array_points_selec[-1]

        mean_array_points_bound_1_rev = mean_array_points_bound_1[::-1]
        list_boundary = list(mean_array_points_bound_1_rev)
        list_boundary = list_boundary + list(mean_array_points_bound_3)
        list_boundary = list_boundary + list(mean_array_points_bound_2)
        mean_array_points_bound_4_rev = mean_array_points_bound_4[::-1]
        list_boundary = list_boundary + list(mean_array_points_bound_4_rev)
        list_boundary = np.array(list_boundary)
    else:
        list_boundary = array_bound_removed

    mask = np.zeros(shape_zero_img, dtype=np.uint8)
    param_contour = np.round(np.array(list_boundary))
    cv.fillPoly(mask, [param_contour.astype(np.int32)], color=1)

    masked_interpolated_img = ma.masked_array(interpolated_img, mask==0)

    return [interpolated_img, list_boundary, masked_img, mask, masked_interpolated_img]


def mean_param(dict_param_maps, dti_map_names, array_points_selec, shape_zero_img, r_row=1, r_col=2):
    import numpy as np

    dict_maps_mean_imgs = {}
    dict_maps_mean_cc_imgs = {}
    dict_maps_min_max = {}

    for j,dti_map in enumerate(dti_map_names):

        map_name = dti_map

        dict_param = dict_param_maps[dti_map]

        tmp_min = np.inf
        tmp_max = -np.inf

        array_param = np.stack(list(dict_param.values()))
        mean_array_param = array_param.mean(axis=0)
        mean_array_param = mean_array_param[r_row:-r_row,r_col:-r_col]

        if mean_array_param.min() < tmp_min:
            tmp_min = mean_array_param.min()
        if mean_array_param.max() > tmp_max:
            tmp_max = mean_array_param.max()

        param_img_cc = map_to_cc(array_points_selec, mean_array_param, shape_zero_img)

        dict_maps_min_max[map_name] = [tmp_min, tmp_max]
        dict_maps_mean_imgs[map_name] = mean_array_param
        dict_maps_mean_cc_imgs[map_name] = param_img_cc

    return dict_maps_mean_imgs, dict_maps_mean_cc_imgs, dict_maps_min_max