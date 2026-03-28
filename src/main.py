import time

from typing import TypeAlias

from PIL import Image
import numpy as np

from .config import FILE_PATH, SCALE, DIST_THRESHOLD


start_time = time.time()


Pixel: TypeAlias = np.typing.NDArray[np.uint8]


class EnhancableImage:
    def __init__(self, *, path=None, img=None):
        if path:
            with Image.open(path) as _img:
                self._matrix = np.array(_img.convert("RGB"))

            return

        if img:
            self._matrix = np.array(img.convert("RGB"))

    def enhance(self, *, scale=2):
        if scale < 2:
            raise ValueError("Scale cannot be less than 2")

        matrix = self._matrix

        for _ in range(2, scale):
            matrix = self._enhance(matrix)

        result_image = Image.fromarray(matrix)

        return result_image

    def _enhance(self, matrix):
        expanded_matrix = np.full(
            (
                matrix.shape[0] * 2 - 1,
                matrix.shape[1] * 2 - 1,
                matrix.shape[2],
            ),
            255,
            dtype=matrix.dtype,
        )

        expanded_matrix[::2, ::2] = matrix

        for row in range(0, expanded_matrix.shape[0] - 2, 2):
            for col in range(0, expanded_matrix.shape[1] - 2, 2):
                self._enhance_pixel_window(expanded_matrix, row, col)

        return expanded_matrix

    def _enhance_pixel_window(self, expanded_matrix, starting_row, starting_col):
        pixel_1 = expanded_matrix[starting_row, starting_col]
        pixel_2 = expanded_matrix[starting_row, starting_col + 2]
        pixel_3 = expanded_matrix[starting_row + 2, starting_col]
        pixel_4 = expanded_matrix[starting_row + 2, starting_col + 2]

        expanded_matrix[starting_row, starting_col + 1] = self._derive_average_pixel(
            pixel_1, pixel_2, pixel_1
        )
        expanded_matrix[starting_row + 1, starting_col] = self._derive_average_pixel(
            pixel_1, pixel_3, pixel_3
        )
        expanded_matrix[starting_row + 1, starting_col + 1] = (
            self._derive_average_pixel(
                self._derive_average_pixel(pixel_1, pixel_2, pixel_2),
                self._derive_average_pixel(pixel_3, pixel_4),
                pixel_3,
            )
        )
        expanded_matrix[starting_row + 1, starting_col + 2] = (
            self._derive_average_pixel(pixel_2, pixel_4, pixel_4)
        )
        expanded_matrix[starting_row + 2, starting_col + 1] = (
            self._derive_average_pixel(pixel_3, pixel_4, pixel_3)
        )

    def _derive_average_pixel(
        self, pixel_1: Pixel, pixel_2: Pixel, default_pixel=None
    ) -> Pixel:
        dist = np.linalg.norm(pixel_1 - pixel_2)

        if dist > DIST_THRESHOLD:
            # preserve sharp edges
            return pixel_1
        else:
            return np.average([pixel_1, pixel_2], axis=0).astype(np.uint8)


enhanced_image = EnhancableImage(path=FILE_PATH).enhance(scale=SCALE)

load_duration = time.time() - start_time

print("Duration: ", load_duration)

enhanced_image.show()
