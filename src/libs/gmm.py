import cv2
import numpy as np
from sklearn.mixture import GaussianMixture

EPS = np.finfo(float).eps


class BayesianColorClassifierLabGMM:
    def __init__(self, n_components=5, covariance_type="full"):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.query_gmm = None
        self.non_query_gmm = None
        self.query_prior = None
        self.non_query_prior = None

    def _extract_lab_pixels(self, images):
        pixels = []
        for img in images:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lab_flat = lab.reshape(-1, 3)
            pixels.append(lab_flat)
        return np.vstack(pixels)

    def fit(self, query_images, non_query_images):
        """
        query_images: list of BGR images belonging to query color class.
        non_query_images: list of BGR images belonging to non-query color class.
        """
        # Extract Lab pixels
        query_pixels = self._extract_lab_pixels(query_images)
        non_query_pixels = self._extract_lab_pixels(non_query_images)

        # Fit GMMs
        self.query_gmm = GaussianMixture(
            n_components=self.n_components, covariance_type=self.covariance_type
        ).fit(query_pixels)

        self.non_query_gmm = GaussianMixture(
            n_components=self.n_components, covariance_type=self.covariance_type
        ).fit(non_query_pixels)

        # Set priors
        total_samples = len(query_images) + len(non_query_images)
        self.query_prior = len(query_images) / total_samples
        self.non_query_prior = len(non_query_images) / total_samples

    def predict(self, bgr_image):
        """
        Given a BGR image, return posterior probabilities for each pixel.
        """
        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        lab_flat = lab.reshape(-1, 3)

        # Compute log probabilities
        assert self.query_gmm is not None and self.non_query_gmm is not None
        log_prob_query = self.query_gmm.score_samples(lab_flat)
        log_prob_non = self.non_query_gmm.score_samples(lab_flat)

        # Convert to probabilities
        p_lab_given_query = np.exp(log_prob_query)
        p_lab_given_non = np.exp(log_prob_non)

        # Compute marginal
        p_lab = (
            p_lab_given_query * self.query_prior
            + p_lab_given_non * self.non_query_prior
        )

        # Avoid division by zero
        p_lab = np.maximum(p_lab, EPS)

        # Compute posterior
        posterior = (p_lab_given_query * self.query_prior) / p_lab

        # Reshape to image shape
        posterior_image = posterior.reshape(bgr_image.shape[:2])

        return posterior_image
