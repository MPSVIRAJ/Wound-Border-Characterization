import cv2
import numpy as np
from scipy.optimize import curve_fit

#--- Z score filter function ---
def zscore_filter(body,depth):
    # Calculate the Z-score for the depth image
    z_score = (depth - depth.mean())/ depth.std()
    # Safty copy of body mask
    body_clensed = body.copy()
    # Correct the body mask according to the threshold of Z-score
    body_clensed[(z_score < -2) | (z_score > 2)] = 0
    # Re-mask the depth map according to the new body mask
    depth_clensed = cv2.bitwise_and(depth, depth, mask=body_clensed)
    # Return the cleaned body mask and depth map
    return body_clensed, depth_clensed

#--- Depth correction for body curvature ---
def quad_surface(xy, a, b, c, d, e, f):
    x, y = xy
    return a + b*x + c*y + d*x**2 + e*y**2 + f*x*y

def depth_corrction_for_body_curvature(wound, body, depth):
    # Define the kernel for morphological operations
    kernel = cv2.getStructuringElement(shape = cv2.MORPH_ELLIPSE, ksize = (20, 20))
    
    # Apply Z-score filter to body and depth
    body_clensed,depth_clensed = zscore_filter(body,depth)
    # Dilate the wound to keep track of a smarter ROI
    dilated_wound = cv2.dilate(wound, kernel = kernel, iterations=15)
    # Remove the original wound from the dilated wound
    dilated_wound = dilated_wound - wound
    # Re-mask the wound using the body mask
    dilated_wound = cv2.bitwise_and(dilated_wound, dilated_wound, mask=body_clensed)   
    # Select the depth values where the dilated area is present. Zero out the rest
    depth_dilated = np.where(dilated_wound == 0, 0, depth_clensed)
    
    # Get the coordinates and values of the non-zero depth pixels
    x, y = np.nonzero(depth_dilated)
    # Extract the corresponding depth values
    z = depth_dilated[depth_dilated != 0]
    # Fit the polynomial surface to the depth data
    popt, pcov = curve_fit(quad_surface, xdata = (x, y), ydata = z)
    # Create a new grid for the curvature correction
    h, w = depth.shape
    y, x = np.mgrid[0:h, 0:w]
    # Calculate the fitted surface
    fitted_surface = quad_surface((y, x), *popt)
    # Subtract the fitted surface from the original depth map to correct for body curvature
    depth_corrected = depth_clensed - fitted_surface
    # Re-mask the corrected depth map using the body mask
    depth_corrected = cv2.bitwise_and(depth_corrected, depth_corrected, mask=body_clensed)
    # Return the cleaned body mask and corrected depth map
    return body_clensed, depth_clensed, depth_corrected
 
#--- Peri-Wound Rectification ---
def sample_pixels_from_contour(img, mask):
    # Getting the contours of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Getting the longest contour for the safety 
    # And removes unnecessary, single-dimension entries from the contour array's shape.
    contour = np.squeeze(max(contours, key=len))
    # Swapping height and width axes of the image to match the contour's shape. 
    # Image is in HWC format, while contour is in WH format.
    pixs = np.swapaxes(img, axis1=0, axis2=1)
    # Sampling pixels from the image using the contour coordinates
    # The contour is a 2D array of coordinates, so we use tuple to index {transpose the array and convert to tuple, 
    # Two separate 1D arrays for x and y coordinates}
    pixs = pixs[tuple(contour.T)]
    # Expanding the dimensions to match the expected shape: like matrix of pixels
    pixs = np.expand_dims(pixs, axis=1)
    return pixs

def unroll_periwound_to_image(img, mask, iterations : int = 1):
    # Getting the base strip of pixels from the contour
    pixl1 = sample_pixels_from_contour(img, mask)
    # Getting the shape of the base strip
    h, w, *_ = pixl1.shape
    # Initializing the list of dilations from the base strip
    dilations = [pixl1]
    # Defining the 3x3 elliptical kernel for dilation
    kernel = cv2.getStructuringElement(shape = cv2.MORPH_ELLIPSE, ksize = (3, 3)) 
    temp_msk = mask.copy()
    Ndilations = 1
    # Dialation process
    for i in range(iterations):
        # Dilating the mask
        temp_msk = cv2.dilate(temp_msk, kernel, iterations=1)
        # Getting the pixels from the dilated mask
        pixl2 = sample_pixels_from_contour(img, temp_msk)       
        # Checking if the plexels are empty
        # If the dilation breaks the mask into several parts, we stop the process
        if pixl2 is None: break
        # Resizing the pixels to the original shape
        # because the dilation may change the shape of the mask
        pixl2 = cv2.resize(pixl2, (w, h), interpolation=cv2.INTER_CUBIC)
        # Appending the dilated pixels to the base strip 
        dilations.append(pixl2)
    num_dilations = len(dilations)
    temp_msk = mask.copy()
    erosions = []
    num_erostions = 1
    # Erosion process
    for i in range(iterations):
        # Eroding the mask
        temp_msk = cv2.erode(temp_msk, kernel, iterations=1)
        # Checking wheather the erosion breaks the mask into several parts
        # Originally there are 2 components background and the mask
        num_components, _ = cv2.connectedComponents(temp_msk)
        if num_components > 2:
            # If the erosion breaks the mask into several parts, we stop the process
            break
        # Getting the pixels from the eroded mask
        pixl2 = sample_pixels_from_contour(img, temp_msk)
        # Resizing the pixels to the original shape
        # Because the erosion may change the shape of the mask
        pixl2 = cv2.resize(pixl2, (w, h), interpolation=cv2.INTER_CUBIC)
        # Appending the eroded pixels to the base strip
        erosions.append(pixl2)
    num_erostions = len(erosions)
    # Concatenating the dilations and erosions
    unrolled_image = np.hstack( erosions[::-1]+dilations)
    
    # Check the number of dimensions of the unrolled image and transpose accordingly
    if unrolled_image.ndim == 3: # For a 3D RGB image
        transposed_image = unrolled_image.transpose(1, 0, 2)
    else: # For a 2D depth map
        transposed_image = unrolled_image.transpose(1, 0)

    return transposed_image, (num_erostions, num_dilations)