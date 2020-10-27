import unittest
from visual_gtsam.barcode_detector import BarcodeDetector
from visual_gtsam.barcode_detector.utils import Net
from visual_gtsam.dataset import Dataset
import cv2

cv2_show = True


class TestDetector(unittest.TestCase):
    def test_detector_initialisation(self):
        detector = BarcodeDetector()
        self.assertIsInstance(detector.get_net(), Net)

    def test_mask_generating(self):
        dataset = Dataset()
        detector = BarcodeDetector()
        image = dataset.get_next_image().get_origin_rgb()
        mask = detector.get_mask(image, thresh=0.95)
        image_with_mask = detector.visualize_mask(image, mask, apply_resize=True)
        if cv2_show:
            cv2.imshow("source_image", image)
            cv2.imshow("generated_mask", mask)
            cv2.imshow("image_with_mask", image_with_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        mask = detector.resize_mask_to_origin(image, mask)
        contours = detector.get_contours(mask, apply_postprocessing=False)
        centers = detector.get_contours_centers(contours)
        image_with_centers = detector.draw_centers_on_image(centers, image)
        if cv2_show:
            cv2.imshow("source_image", image)
            cv2.imshow("generated_mask", mask)
            cv2.imshow("image_with_centers", image_with_centers)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
