import cv2
import numpy as np
import numpy


class SkinColorFilter:
    """
    This class implements a number of functions to perform skin color filtering.

    It is based on the work published in "Adaptive skin segmentation via feature-based face detection",
    M.J. Taylor and T. Morris, Proc SPIE Photonics Europe, 2014 [taylor-spie-2014]_

    **Attributes:**
      ``mean`` : (numpy array 2x1)
        the mean skin color

      ``covariance`` : (numpy array 2x2)
        the covariance matrix of the skin color

      ``covariance_inverse`` : (numpy array 2x2)
        the inverse covariance matrix of the skin color

      ``circular_mask`` : (numpy logical array)
        mask of the size of the image, defining a circular region in the center

      ``luma_mask`` : (numpy logical array)
        mask of the size of the image, defining valid luma values
    """

    def __init__(self):
        self.mean = numpy.array([0.0, 0.0])
        self.covariance = numpy.zeros((2, 2), 'float64')
        self.covariance_inverse = numpy.zeros((2, 2), 'float64')

    def __generate_circular_mask(self, image, radius_ratio=0.4):
        """__generate_circular_mask(image, [radius_ratio]

        This function will generate a circular mask to be applied to the image.

        The mask will be true for the pixels contained in a circle centered in the image center,
        and with radius equals to radius_ratio * the image's height.

        **Parameters:**

          ``image`` : (numpy array)
            The face image.

          ``radius_ratio`` (Optional[float]):
            the ratio of the image's height to define the radius of the circular region.
            Defaults to 0.4.
        """
        # image C H W
        x_center = image.shape[1] / 2
        y_center = image.shape[2] / 2

        # arrays with the image coordinates
        x = numpy.zeros((image.shape[1], image.shape[2]))
        x[:] = range(0, x.shape[1])
        y = numpy.zeros((image.shape[2], image.shape[1]))
        y[:] = range(0, y.shape[1])
        y = numpy.transpose(y)

        # translate s.t. the center is the origin
        x -= x_center
        y -= y_center

        # condition to be inside of a circle: x^2 + y^2 < r^2
        radius = radius_ratio * image.shape[2]
        self.circular_mask = (x ** 2 + y ** 2) < (radius ** 2)

    def __remove_luma(self, image):
        """__remove_luma(image)

        This function remove pixels with extreme luma values.

        Some pixels are considered as non-skin if their intensity is either too high or too low.
        The luma value for all pixels inside a provided circular mask is calculated. Pixels for which the
        luma value deviates more than 1.5 * standard deviation are pruned.

        **Parameters:**

          ``image`` : (numpy array)
            The face image.
        """
        # compute the mean and std of luma values on non-masked pixels only
        luma = 0.299 * image[0, self.circular_mask] + 0.587 * image[1, self.circular_mask] + 0.114 * image[
            2, self.circular_mask]
        m = numpy.mean(luma)
        s = numpy.std(luma)

        # apply the filtering to the whole image to get the luma mask
        luma = 0.299 * image[0, :, :] + 0.587 * image[1, :, :] + 0.114 * image[2, :, :]
        self.luma_mask = numpy.logical_and((luma > (m - 1.5 * s)), (luma < (m + 1.5 * s)))

    def estimate_gaussian_parameters(self, image, pre_mask=None):
        """estimate_gaussian_parameters(image)

        This function estimates the parameter of the skin color distribution.

        The mean and covariance matrix of the skin pixels in the normalised rg colorspace are computed.
        Note that only the pixels for which both the circular and the luma mask is 'True' are considered.

        **Parameters:**

          ``image`` : (numpy array)
            The face image.
        """
        self.__generate_circular_mask(image)
        self.__remove_luma(image)
        mask = numpy.logical_and(self.luma_mask, self.circular_mask)
        if pre_mask is not None:
            mask = numpy.logical_and(mask, pre_mask)

        # get the mean
        channel_sum = image[0].astype('float64') + image[1] + image[2]
        nonzero_mask = numpy.logical_or(numpy.logical_or(image[0] > 0, image[1] > 0), image[2] > 0)
        r = numpy.zeros((image.shape[1], image.shape[2]))
        r[nonzero_mask] = image[0, nonzero_mask] / channel_sum[nonzero_mask]
        g = numpy.zeros((image.shape[1], image.shape[2]))
        g[nonzero_mask] = image[1, nonzero_mask] / channel_sum[nonzero_mask]
        self.mean = numpy.array([numpy.mean(r[mask]), numpy.mean(g[mask])])

        # get the covariance
        r_minus_mean = r[mask] - self.mean[0]
        g_minus_mean = g[mask] - self.mean[1]
        samples = numpy.vstack((r_minus_mean, g_minus_mean))
        samples = samples.T
        cov = sum([numpy.outer(s, s) for s in samples])
        self.covariance = cov / float(samples.shape[0] - 1)

        # store the inverse covariance matrix (no need to recompute)
        if numpy.linalg.det(self.covariance) != 0:
            self.covariance_inverse = numpy.linalg.inv(self.covariance)
        else:
            self.covariance_inverse = numpy.zeros_like(self.covariance)

    def get_skin_mask(self, image, threshold, pre_mask=None):
        """get_skin_mask(image, [threshold]) -> skin_mask
        This function computes the probability of skin-color for each pixel in the image.

        **Parameters:**

          ``image`` : (numpy array)
            The face image.

          ``threshold`` : (Optional, float between 0 and 1)
            the threshold on the skin color probability. Defaults to 0.5

        **Returns:**

          ``skin_mask`` : (numpy logical array)
          The mask where skin color pixels are labeled as True.
        """
        # skin_map = numpy.zeros((image.shape[1], image.shape[2]), 'float64')

        # get the image in rg colorspace
        channel_sum = image[0].astype('float64') + image[1] + image[2]
        nonzero_mask = numpy.logical_or(numpy.logical_or(image[0] > 0, image[1] > 0), image[2] > 0)
        r = numpy.zeros((image.shape[1], image.shape[2]), 'float64')
        r[nonzero_mask] = image[0, nonzero_mask] / channel_sum[nonzero_mask]
        g = numpy.zeros((image.shape[1], image.shape[2]), 'float64')
        g[nonzero_mask] = image[1, nonzero_mask] / channel_sum[nonzero_mask]

        # compute the skin probability map
        r_minus_mean = r - self.mean[0]
        g_minus_mean = g - self.mean[1]
        v = numpy.dstack((r_minus_mean, g_minus_mean))
        v = v.reshape((r.shape[0] * r.shape[1], 2))
        probs = [numpy.dot(k, numpy.dot(self.covariance_inverse, k)) for k in v]
        probs = numpy.array(probs).reshape(r.shape)
        skin_map = numpy.exp(-0.5 * probs)

        skin_mask = skin_map > threshold
        if pre_mask is not None:
            skin_mask = np.logical_and(skin_mask, pre_mask)
        return skin_mask


def detect_skin_bob(img_bgr, threshold=0.1, pre_mask=None):
    img_rgb = img_bgr[:, :, ::-1]
    img_rgb = np.transpose(img_rgb, [2, 0, 1, ])
    skin_filter = SkinColorFilter()
    skin_filter.estimate_gaussian_parameters(img_rgb, pre_mask=pre_mask)
    skin_mask = skin_filter.get_skin_mask(img_rgb, threshold, pre_mask=pre_mask)
    return skin_mask

