import scipy.io
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Select specific bands (for example, first three bands for RGB-like visualization)
def plot_image(data, title, bands=[0, 1, 2]):
    # Select the desired bands for RGB visualization
    image = data[:, :, bands]  # Selecting the first 3 bands as an RGB image
    image = np.clip(image, 0, 1)  # Ensure the values are between 0 and 1 for display
    
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()


data = scipy.io.loadmat('./mcodes/dataset/Chikusei_x4/evals/block_Chikusei_train_3347.mat')

# View keys (variables stored inside)
print(data.keys())

# Access specific data
ms = data['ms']  # Multi-spectral patch
lms = data['ms_bicubic']  # Low-res upsampled version
gt = data['gt']  # Ground truth hyperspectral patch

def mat_to_tiff(hyperspectral_data, output_tiff_path):
   
    height, width,bands = hyperspectral_data.shape
    
    # Define metadata for the GeoTIFF
    transform = rasterio.transform.from_bounds(
        west=0, south=0, east=width, north=height, width=width, height=height
    )
    
    # Create the GeoTIFF file
    with rasterio.open(
        output_tiff_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=bands,
        dtype=hyperspectral_data.dtype,
        transform=transform,
        crs='+proj=latlong'
    ) as dst:
        # Write the data
        for band in range(bands):
            dst.write(hyperspectral_data[ :, :,band], band + 1)

# Example usage
tiff_file = 'output_file.tiff'
mat_to_tiff(gt, tiff_file)

# Shape of the tensors
print(ms.shape, lms.shape, gt.shape)
# Visualize the three patches (ms, lms, gt)
# plot_image(ms, "Multi-Spectral Patch (ms)")
# plot_image(lms, "Low-Res Upsampled Patch (lms)")
# plot_image(gt, "Ground Truth Patch (gt)")