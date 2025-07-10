import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit    
from scipy.stats import skew, kurtosis
from feature_extraction import linear_func, sigmoid_func

# Plot initial data
def plot_initial_data(image, wound_mask, body_mask, depth_map):
    """
    Visualizes the raw input data: the RGB image with contours and the depth map.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(image)
    ax1.grid(False)
    if wound_mask is not None:
        ax1.contour(wound_mask, colors='red', linewidths=2, levels=[0.5])
    if body_mask is not None:
        ax1.contour(body_mask, colors='blue', linewidths=1, levels=[0.5])
    ax1.set_title("RGB Image with Wound and Body Masks")
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.imshow(np.where(depth_map > 0, depth_map, np.nan), cmap='plasma_r')
    ax2.set_title("Depth Map with Body Mask Applied")
    ax2.grid(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    plt.tight_layout()
    plt.show()

# Visualize rectified depth strip of 
def show_unrolled_strip(rect_depth, unrolled_image, d1, p1, iterations):
    """    
    Visualizes the unrolled depth map and RGB image with 
    a horizontal line at the wound border position.
    """
    rect_depth_for_plot = rect_depth.transpose(1, 0)
    unroled_image = unrolled_image.transpose(1, 0, 2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, iterations/20))
    ax1.imshow(rect_depth_for_plot, cmap='plasma_r')
    ax1.axhline(d1, linestyle='dashed', color='w', linewidth=2)
    ax1.set_title("Unrolled Depth Map")

    ax2.imshow(unroled_image)
    ax2.axhline(p1, linestyle='dashed', color='w', linewidth=2)
    ax2.set_title("Unrolled RGB Image")
    ax1.grid(False)  # <--- Explicitly turn off grid
    ax2.grid(False)  # <--- Explicitly turn off grid    


# Visualize the mean depth profile with its standard deviation
def plot_depth_profile(mean_profile, std_profile, d1):
    """
    Visualizes the mean depth profile with its standard deviation and the
    estimated wound border position.
    """
    plt.figure(figsize=(15, 7))
    # Plot the mean profile line
    plt.plot(mean_profile, color='blue', linewidth=1, alpha=0.9, label='Mean Profile')  
    # Add the standard deviation as a shaded area
    plt.fill_between(range(len(mean_profile)),
                     mean_profile - std_profile,
                     mean_profile + std_profile,
                     color='lightblue', alpha=0.6, label='Standard Deviation')

    # Add a vertical line for the wound border
    plt.axvline(x=d1, color='red', linestyle='--', linewidth=2, label=f'Wound Border Position (d1={d1})')
    
    plt.title('Mean Depth Profile of Periwound Area', fontsize=16)
    plt.xlabel('Profile Position (Wound Interior -> Surrounding Skin)', fontsize=12)
    plt.ylabel('Mean Corrected Depth', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


# Visualizing feature fits on the mean depth profile
def plot_profiles_and_fits(mean_profile, std_profile, smoothed_profile, features, d1, p1, transition_width, save):
    """
    Plots the mean and std deviation profiles, the smoothed profile, and the curve fits.
    """
    plt.figure(figsize=(15, 7))
    
    x = np.arange(len(mean_profile))
    
    # Plot the original mean profile (noisy)
    #plt.plot(x, mean_profile,'o', color='lightgreen', alpha=0.5, label='Original Mean Profile')
    
    # Plot the smoothed profile
    plt.plot(x, smoothed_profile, color='gray', 
             linewidth=2, 
             label='Smoothed Mean Profile')
    
    # Plot the standard deviation as a shaded area around the smoothed profile
    plt.fill_between(x, smoothed_profile - std_profile, 
                     smoothed_profile + std_profile, 
                     color='lightblue', alpha=0.5, 
                     label='Standard Deviation')

    plt.axvline(x=d1, color='red', 
                linestyle='--', linewidth=2, 
                label=f'Wound Edge (baseline contour)')

    # Define regions for plotting fits
    end_bed = d1 - transition_width // 2
    x_bed = np.arange(0, end_bed)
    start_edge = end_bed
    end_edge = min(len(mean_profile), d1 + transition_width // 2)
    x_edge = np.arange(start_edge, end_edge)
    start_skin = end_edge
    x_skin = np.arange(start_skin, len(mean_profile))

    # Plot fits if they were successful
    if features.get('bed_fit_success'):
        params_bed = [features['bed_slope'], features['bed_intercept']]
        plt.plot(x_bed, linear_func(x_bed, *params_bed), color='blue', 
                 linewidth=3, label="Wound Bed Fit")

    if features.get('edge_fit_success'):
        params_edge = [features['edge_amplitude'], features['edge_steepness'], features['edge_midpoint'], features['edge_offset']]
        plt.plot(x_edge, sigmoid_func(x_edge, *params_edge), 
                 color='green', linewidth=3, 
                 label="Edge Sigmoid Fit")

    if features.get('skin_fit_success'):
        params_skin = [features['skin_slope'], features['skin_intercept']]
        plt.plot(x_skin, linear_func(x_skin, *params_skin), 
                 color='purple', linewidth=3, 
                 label="Healthy Skin Fit")

    plt.title('Mean Depth Profile with Piecewise Curve Fitting',fontsize=16)
    plt.xlabel('Width of the rectified strip',fontsize=14)
    plt.ylabel('Mean depth',fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(False)
    plt.xlim(20, 170)
    plt.show()
    if save:
        plt.savefig(
            '../report/fig/mean_depth_cf.png',
            dpi=300,
            bbox_inches='tight' ) 