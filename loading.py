
import warnings
from typing import Optional

import mmengine.fileio as fileio
import numpy as np

import mmcv
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.builder import TRANSFORMS

import math


@TRANSFORMS.register_module()
class CSILoadImageFromFile(BaseTransform):
    """Load an image or process `.npy` data with center cropping.

    Required Keys:
    - img_path

    Modified Keys:
    - img
    - img_shape
    - ori_shape

    Optional Keys for `.npy`:
    - frame (for center cropping)

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. Defaults to False.
        color_type (str): Image color type for `mmcv.imfrombytes`. Defaults to 'color'.
        imdecode_backend (str): Backend for image decoding. Defaults to 'cv2'.
        file_client_args (dict, optional): Deprecated in favor of `backend_args`.
        ignore_empty (bool): Allow loading empty image or nonexistent file path. Defaults to False.
        backend_args (dict, optional): Instantiates a corresponding file backend.
        crop_length (int, optional): Length of the crop for `.npy` files. Defaults to None.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 crop_length: Optional[int] = None,
                 *,
                 backend_args: Optional[dict] = None) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.crop_length = crop_length

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Function to load an image or `.npy` data with optional center cropping.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image or processed `.npy` data.
        """
        filename = results['img_path']
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                file_data = file_client.get(filename)
            else:
                file_data = fileio.get(
                    filename, backend_args=self.backend_args)

            if filename.endswith('.npy'):  # Process `.npy` files
                data = np.load(filename)  # Load the `.npy` file

                # Ensure the crop_length is defined
                if self.crop_length is not None and 'frame' in results:
                    frame = results['frame']  # Current frame
                    total_pkgs = data.shape[0]  # Total number of packages
                    num_portions = 90 - 1  # Divide into 89 portions
                    portion_size = total_pkgs / num_portions  # Size of one portion

                    # Find the center index for the current frame
                    center_idx = round((frame - 1) * portion_size)
                    half_crop = self.crop_length // 2  # Half of the crop length
                    start_idx = max(0, center_idx - half_crop)  # Start of the crop
                    end_idx = min(total_pkgs, center_idx + half_crop)  # End of the crop

                    # Adjust boundaries to ensure exact crop_length
                    if end_idx - start_idx < self.crop_length:
                        if start_idx == 0:
                            end_idx = start_idx + self.crop_length
                        elif end_idx == total_pkgs:
                            start_idx = end_idx - self.crop_length

                    # Perform cropping
                    data = data[start_idx:end_idx]

                results['img'] = data
                results['img_shape'] = (1080, 1920)
                results['ori_shape'] = (1080, 1920)

            else:  # Process image files
            # NOTE: It should never enter this case 
                img = mmcv.imfrombytes(
                    file_data, flag=self.color_type, backend=self.imdecode_backend)
                assert img is not None, f'failed to load image: {filename}'

                if self.to_float32:
                    img = img.astype(np.float32)

                results['img'] = img
                results['img_shape'] = data.shape[:2]
                results['ori_shape'] = data.shape[:2]


        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', "
                    f"crop_length={self.crop_length}, ")

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str
    
