from skimage import morphology


# Removal of areas with small connectivity domains
def denoise_small_objects(input_array):
    input_array = morphology.remove_small_objects(input_array.astype(bool), min_size=300, connectivity=1)
    # print(np.issubdtype(input_array.dtype, np.integer))
    output_array = input_array.astype('int8')
    return output_array
