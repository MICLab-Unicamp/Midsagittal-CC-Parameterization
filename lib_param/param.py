
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

    #-------------------------------------
    # Finding idx of the anterior endpoint
    #-------------------------------------

    if max_y_anterior_index > max_x_low_anterior_index:
        print("Warning! Problem with the definition of the anterior endpoint!")

    max_curv_idx = np.argmax(curvature[idxs[0]:idxs[1]][max_y_anterior_index:max_x_low_anterior_index]) + (max_y_anterior_index)

    #--------------------------------------
    # Finding idx of the posterior endpoint
    #--------------------------------------

    # Identifying the posterior half of the boundary
    posterior_indices = np.where(new_xInter > np.mean(new_xInter))
    # Identifying the index of the maximum y-coordinate considering the posterior half of the boundary
    post_end_idx = posterior_indices[0][np.argmax(new_yInter[posterior_indices])]

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
    spline,_ = interpolate.make_splprep(np.array([lower_bound_y,lower_bound_x]),s=0)
    yInter_low,xInter_low = spline(unew)
    spline,_ = interpolate.make_splprep(np.array([upper_bound_y,upper_bound_x]),s=0)
    yInter_up,xInter_up = spline(unew)

    # Getting the centerline points from the upper and lower points
    xInter_center = (xInter_low+xInter_up)/2
    yInter_center = (yInter_low+yInter_up)/2

    return xInter_low, yInter_low, xInter_up, yInter_up, xInter_center, yInter_center, new_xInter, new_yInter


# Parameterization points definition
def param_points(sub_path, cc_msp_fname, np_bound=225, np_transv=25, s_fac=1):
    import numpy as np
    import os
    from dipy.io.image import load_nifti

    # Loading midsagittal CC mask
    cc_msp,_ = load_nifti(os.path.join(sub_path, cc_msp_fname))

    # Selecting only the midsagittal slice and reorienting the 2D image to obtain the CC boundaries
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
def param(sub_path, cc_msp_fname, dti_map_info, np_bound=225, np_transv=25, s_fac=1):
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
