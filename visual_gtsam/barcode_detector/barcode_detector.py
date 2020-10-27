import cv2
import numpy as np
import torch
import os
from visual_gtsam.barcode_detector.utils import Net
import torchvision.transforms.functional as TF
from google_drive_downloader import GoogleDriveDownloader as gdd


class BarcodeDetector:
    _net = None
    _device = ""
    def __init__(self, main_dir="downloads", file_id="1siCjXOFcRXY4FG9X2xZ61hG3Oh_DNCAn", path_name="net_weights.pth"):
        self._main_dir = main_dir
        self._file_id = file_id
        self._path_name = path_name
        self._is_downloaded = False

        self._kernel_size = 9
        self._postprocessing_kernel = np.ones(self._kernel_size)

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = Net()
        path_to_weights = "./{0}/{1}".format(main_dir, path_name)
        self._download_weights()

        if torch.cuda.is_available():
            print("Launching from CUDA")
            checkpoint = torch.load(f=path_to_weights)
        else:
            print("Launching from CPU")
            checkpoint = torch.load(map_location='cpu', f=path_to_weights)
        net.load_state_dict(checkpoint)
        self._net = net.to(self._device)

    def _download_weights(self):
        path_to_weights = "./{0}/{1}".format(self._main_dir, self._path_name)
        if not os.path.exists(path_to_weights):
            print("Loading weights")
            os.makedirs(self._main_dir, exist_ok=True)
            gdd.download_file_from_google_drive(file_id=self._file_id,
                                                dest_path='{}/weights.zip'.format(self._main_dir),
                                                unzip=True)
            os.remove('./{}/weights.zip'.format(self._main_dir))
            self._is_downloaded = True
            print("Weights were downloaded")
        else:
            self._is_downloaded = True
            print("Weights are ready")

    def get_net(self):
        return self._net

    def get_mask(self, image, thresh):
        image = cv2.resize(image, (512, 512))
        image_device = TF.to_tensor(image).float().to(self._device).unsqueeze_(0)
        return np.array(self._net.eval_predict(image_device) > thresh, dtype="uint8") * 255

    @staticmethod
    def resize_mask_to_origin(image, mask):
        origin_dims = image.shape
        return cv2.resize(mask, origin_dims[::-1][-2:])

    def get_contours(self, mask, apply_postprocessing=True):
        if apply_postprocessing:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._postprocessing_kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._postprocessing_kernel)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    @staticmethod
    def get_contours_centers(contours):
        centers = []
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            centers.append([x + w / 2, y + h / 2])
        return centers

    @staticmethod
    def draw_centers_on_image(centers, image_origin, radius=5, color=(0, 0, 255)):
        image = image_origin.copy()
        centers = np.array(centers, dtype=int)
        for center in centers:
            image = cv2.circle(image, tuple(center), radius, color, -1)
        return image

    def visualize_mask(self, image, mask, apply_resize=False):
        """
        Apply your mask to source image
        :param image: Your source image
        :param mask: Your mask
        :param apply_resize: (default - False) Change to resize mask
        :return: visualized image with mask
        """
        b, g, r = cv2.split(image)
        if apply_resize:
            mask = self.resize_mask_to_origin(image, mask)
        color_channeled = cv2.merge(
            (b, np.full(b.shape, 255, dtype=np.uint8) - mask, np.full(b.shape, 255, dtype=np.uint8) - mask))
        return cv2.bitwise_and(image, color_channeled)
