"""Hard morphology requirements for the FDM/cast prototype assembly.

These checks constrain geometry generation, not a claim of magnetic performance.
Candidates must retain both features; performance ranking comes afterwards.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class OrganicPrototypeSpec:
    honeycomb_cell_m: float = .006
    honeycomb_wall_m: float = .0026
    helix_turns: float = 4.0
    helix_radius_m: float = .0015
    core_process: str = "FDM mold + resin/Fe3O4 casting; uncharacterized material"

    def validate(self, cfg, honeycomb, helix):
        if self.honeycomb_wall_m < 2 * max(cfg.spacing):
            raise ValueError("蜂窝壁不足两个体素；提高几何分辨率或增加壁厚")
        if not 0 < self.honeycomb_wall_m < self.honeycomb_cell_m:
            raise ValueError("蜂窝必须保留孔隙，壁厚应小于单元尺寸")
        if self.helix_turns < 1 or self.helix_radius_m <= 0:
            raise ValueError("原型必须包含至少一圈螺旋通道")
        if not np.any(honeycomb.sdf < 0) or not np.any(helix.sdf < 0):
            raise ValueError("蜂窝与螺旋均为必选特征，不能输出空特征")
        return {"required": ["honeycomb", "helix"], "geometry_presence_passed": True,
                "core_process": self.core_process, "material_characterized": False,
                "manufacturing_release": False}
