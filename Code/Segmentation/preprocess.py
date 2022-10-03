import os
import mitk
import nibabel as nib
import numpy as np
import cv2

# from scipy.ndimage import interpolation


class PreProcessing:
    def __init__(self,
                 ornt,
                 spacing,
                 shape,
                 data_type,
                 apply_histeq=False,
                 apply_N4=False,
                 apply_standardization=True,
                 apply_normalization=True):
        self.ornt = ornt
        self.spacing = spacing
        self.shape = shape
        self.apply_histeq = apply_histeq
        self.apply_N4 = apply_N4
        self.apply_standardization = apply_standardization
        self.apply_normalization = apply_normalization
        self.data_type = data_type

    def geometric_transformation(self, image_nii: nib.Nifti1Image, *args, **kwargs):
        # Orientation Calibration
        reoriented_nii = mitk.nii_reorientation(image_nii, end_ornt=self.ornt)
        # Size Calibration
        resized_nii = mitk.nii_resize(reoriented_nii,
                                      target_shape=self.shape,
                                      target_spacing=self.spacing,
                                      *args,
                                      **kwargs)
        return resized_nii

    @staticmethod
    def N4(image_nii: nib.Nifti1Image):
        return mitk.npy_correct_bias_N4(image_nii)

    @staticmethod
    def histeq(array: np.ndarray, *args, **kwargs):
        array = mitk.npy_histogram_equalization(array, *args, **kwargs)
        return array

    @staticmethod
    def standardization(array: np.ndarray):
        mean = np.mean(array)
        std = np.std(array)
        array = (array - mean) / (std + 0.00001)
        return array

    @staticmethod
    def normalization(array: np.ndarray):
        maxv = np.max(array)
        minv = np.min(array)
        array = (array - minv) / (maxv - minv + 0.00001)
        return array

    def processing_image(self, image_nii):
        image_nii = self.geometric_transformation(image_nii)  #, interpolation='nearest')

        # Determine whether to perform bias field correction N4
        if self.apply_N4:
            image_nii = self.N4(image_nii)

        array = image_nii.get_fdata()
        affine = image_nii.affine

        # # Whether to remove the liver portion separately and erode part of the margin (MR fatty)
        # if label_profile_nii:
        #     label_profile_nii = self.geometric_transformation(label_profile_nii, interpolation='nearest')
        #     label_profile_data = label_profile_nii.get_data()
        #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        #     mask = cv2.erode(label_profile_data, kernel)
        #     array = array * mask

        # Take the value from 1% to 99.9%
        # The Dixon sequence vessels are dark. And to improve the contrast of the vessels in the liver, the overall signal value needs to be reduced
        if self.data_type == 'dce':
            percentile_99 = np.percentile(array, 99.9)
            percentile_1 = np.percentile(array, 99.9) // 10
            array[array > percentile_99] = percentile_99
            array[array < percentile_1] = percentile_1
        if self.data_type == 'chest':
            center = 35
            width = 350
            min = (2 * center - width) / 2.0 + 0.5
            max = (2 * center + width) / 2.0 + 0.5
            array[array < min] = min
            array[array > max] = max


        # Whether histogram equalization
        if self.apply_histeq:
            array = self.histeq(array)
        # Whether to standardize
        if self.apply_standardization:
            array = self.standardization(array)
        # Normalized or not
        if self.apply_normalization:
            array = self.normalization(array)

        #### The following code needs to be turned on for the blood vessels (MR fatty) ###
        # array = 1 - array
        # array[array == 1] = 0
        # new_nii = nib.Nifti1Image(array, affine)
        # return new_nii

        new_nii = nib.Nifti1Image(array, affine)
        return new_nii