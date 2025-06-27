import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit    
from scipy.stats import skew, kurtosis
# visualize rectified depth strip of 
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
def plot_feature_fits(mean_profile, d1, features, transition_width=40):
    """
    Visualizes the piecewise fits on top of the mean depth profile.
    This function is for use in exploratory notebooks.
    """
    plt.figure(figsize=(16, 8))
    plt.plot(mean_profile, 'o', color='gray', alpha=0.4, markersize=4, label='Smoothed Profile Data')

    # Define regions again for plotting
    end_bed = max(0, d1 - transition_width // 2)
    x_bed = np.arange(0, end_bed)
    start_edge = end_bed
    end_edge = min(len(mean_profile), d1 + transition_width // 2)
    x_edge = np.arange(start_edge, end_edge)
    start_skin = end_edge
    x_skin = np.arange(start_skin, len(mean_profile))

    # Plot fits if they were successful
    if features.get('bed_fit_success'):
        params_bed = [features['bed_slope'], features['bed_intercept']]
        plt.plot(x_bed, linear_func(x_bed, *params_bed), color='blue', linewidth=3, label=f"Wound Bed Fit (R²={features.get('bed_r2', 0):.2f})")

    if features.get('edge_fit_success'):
        params_edge = [features['edge_amplitude'], features['edge_steepness'], features['edge_midpoint'], features['edge_offset']]
        plt.plot(x_edge, sigmoid_func(x_edge, *params_edge), color='green', linewidth=3, label=f"Edge Sigmoid Fit (R²={features.get('edge_r2', 0):.2f})")

    if features.get('skin_fit_success'):
        params_skin = [features['skin_slope'], features['skin_intercept']]
        plt.plot(x_skin, linear_func(x_skin, *params_skin), color='purple', linewidth=3, label=f"Healthy Skin Fit (R²={features.get('skin_r2', 0):.2f})")

    plt.axvline(x=d1, color='red', linestyle='--', linewidth=2, label='Wound Border Position')
    plt.title('Piecewise Mathematical Fit of Depth Profile', fontsize=16)
    plt.xlabel('Profile Position (Wound Interior -> Surrounding Skin)', fontsize=12)
    plt.ylabel('Mean Corrected Depth', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()