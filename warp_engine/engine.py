from .template import load_template
from .global_homography import warp_single_h
from .idw_refine import idw_refine
from .region_warp import refine_regions
from .config import A4_PX

class WarpEngine:

    def __init__(self, template_path):
        self.template = load_template(template_path)

    def warp(self, img, out_size=A4_PX, debug_dir=None,
             use_global_idw=True, use_region_refine=True):

        warped = warp_single_h(img, self.template, out_size, debug_dir)

        if use_global_idw:
            warped = idw_refine(warped, self.template, debug_dir=debug_dir)

        if use_region_refine:
            warped = refine_regions(warped, self.template, debug_dir=debug_dir)

        return warped
