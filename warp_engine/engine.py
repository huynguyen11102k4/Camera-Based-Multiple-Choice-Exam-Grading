import cv2 as cv
import numpy as np

from .template import load_template
from .global_homography import compute_global_h
from .idw_refine import idw_refine
from .region_warp import refine_regions
from .config import A4_PX


class WarpEngine:

    def __init__(self, template_layout_path, template_image_path):
        self.layout = load_template(template_layout_path)
        self.template_img = cv.imread(template_image_path)
        if self.template_img is None:
            raise FileNotFoundError(template_image_path)

        self.template_img = cv.resize(self.template_img, A4_PX)

    def warp(
            self,
            img,
            out_size=A4_PX,
            debug_dir=None,
            use_global_idw=True,
            use_region_refine=True,
    ):
        H, _ = compute_global_h(img, self.layout)

        warped_src = cv.warpPerspective(img, H, out_size)

        if use_global_idw:
            warped_src = idw_refine(
                warped_src,
                self.layout,
                debug_dir=debug_dir,
            )

        base = self.template_img.copy()

        if use_region_refine:
            base = refine_regions(
                template_img=base,
                layout=self.layout,
                warped_src=warped_src,
                debug_dir=debug_dir,
            )

        return base
