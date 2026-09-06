"""Artifact-backed part inspection. No inferred cut is a manufacturing joint.

Meshes retain the source material and coordinates. STL copies use millimetres;
GLB uses metres. This is a fit-check package, not a manufacturing release.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import trimesh

from organic_motor.geometry.export import material_mesh
from organic_motor.geometry.voxel import VoxelVolume

REVISION = "parts-v1"
RECIPES = {
    "rotor_iron": ("转子磁轭", "加工 / 采购", "需确认磁性材料、轴连接与磁钢保持结构；不可直接用树脂件替代。"),
    "stator_iron": ("定子磁芯", "磁芯工艺待定", "现有磁芯未定义可绕线拆分接缝。选叠片/合格软磁材料，或先测复合材料 B-H、损耗与温升，再重新求解。"),
    "support_iron": ("蜂窝支撑", "打印配合样件", "蜂窝为必选形态。可打印尺寸样件；改用聚合物后必须更新结构、热与电磁材料，不能沿用铁参数。"),
    "housing_iron": ("螺旋流道外壳", "打印配合样件", "先检查剖面与流道密封。封闭流道尚无可拆密封接缝；STL 仅用于尺寸检查。"),
    "front_endcap_iron": ("前端盖", "打印配合样件", "轴承座尺寸、止口、孔配合需实测与校准；尚未验证螺栓连接及轴承载荷。"),
    "rear_endcap_iron": ("后端盖", "打印配合样件", "检查出线空间、轴承座和装配顺序；打印后验证配合。"),
    "copper": ("绕组", "漆包铜线绕制", "几何不是铜线采购规格。需确定线径、匝数、串并联、引出线和绕线骨架；禁止把此 STL 当作普通打印件。"),
    "rotor_pm": ("永磁体", "采购 / 定制", "需确定牌号、磁化方向、尺寸公差与保持方式，树脂不是永磁体。"),
    "insulator": ("绕组绝缘", "绝缘工艺待定", "根据线径、温度等级与爬电间隙设计骨架；普通树脂参数未验证。"),
    "coolant": ("螺旋流道 · 空腔", "空腔参考 · 不打印", "蓝色体积代表流体空间，不是实体零件。检查端口连通、封闭壁和清洗路径。"),
    "iron": ("未分件磁性实体", "缺少零件身份", "旧模型没有零件掩码。重新生成 assembly；不能凭颜色判断可制造性。"),
}


def part_meshes(npz_path: Path, level=0.35, smoothing="none", iterations=0, view="full", materials=None):
    """Partition by explicit source masks, without changing the parent solid."""
    from organic_motor.web.builder import _load_volume, apply_view
    vol = apply_view(_load_volume(npz_path), view)
    with np.load(npz_path, allow_pickle=False) as data:
        masks = {k: np.asarray(data[k], dtype=np.float32) for k in
                 ("rotor_mask", "support_mask", "housing_mask", "endcap_mask") if k in data.files}
    requested = set(materials or vol.materials)
    fields = []
    if "iron" in requested:
        remaining = vol.iron.copy()
        z = vol.origin[2] + np.arange(vol.shape[2]) * vol.spacing[2]
        mid = (z[0] + z[-1]) / 2
        # End caps have first priority: never get merged into yoke/support.
        regions = []
        if "endcap_mask" in masks:
            regions.extend([
                ("front_endcap_iron", masks["endcap_mask"] * (z[None,None,:] >= mid)),
                ("rear_endcap_iron", masks["endcap_mask"] * (z[None,None,:] < mid)),
            ])
        regions.extend((n, masks[k]) for n,k in (
            ("rotor_iron", "rotor_mask"), ("support_iron", "support_mask"),
            ("housing_iron", "housing_mask")) if k in masks)
        for name, mask in regions:
            mask = np.clip(mask, 0, 1)
            fields.append((name, "iron", remaining * mask))
            remaining = remaining * (1-mask)
        fields.append(("stator_iron" if masks else "iron", "iron", remaining))
    for mat in ("copper", "pm", "insulator", "coolant"):
        if mat in requested and mat in vol.materials:
            fields.append(("rotor_pm" if mat == "pm" else mat, mat, vol.materials[mat]))
    for name, mat, density in fields:
        if density.max(initial=0) <= level:
            continue
        sub = VoxelVolume(iron=density, pm=np.zeros_like(density), spacing=vol.spacing, origin=vol.origin)
        mesh = material_mesh(sub, "iron", level=level, smoothing=smoothing, smoothing_iterations=iterations)
        if mesh is None:
            continue
        if mesh.volume < 0:
            mesh.invert()
        label, process, note = RECIPES[name]
        mesh.metadata.update(part_id=name, label=label, material=mat,
                             motion="rotor" if name.startswith("rotor_") else "stator",
                             process=process, note=note)
        yield name, mesh


def describe(name, mesh):
    label, process, note = RECIPES[name]
    # Connectivity is evidence; disconnected coils remain a winding assembly,
    # rather than inventing separate manufacturable pieces from voxel fragments.
    components = len(mesh.split(only_watertight=False))
    return dict(id=name, label=label, process=process, note=note,
                size_mm=np.round(mesh.extents * 1000, 2).tolist(),
                center_m=mesh.bounds.mean(axis=0).tolist(),
                watertight=bool(mesh.is_watertight), components=components,
                volume_mm3=round(abs(float(mesh.volume))*1e9, 2),
                printable_release=False, role="void" if name == "coolant" else "solid")


def manifest(npz_path, meshes):
    parts = [describe(n,m) for n,m in meshes]
    names = {p["id"] for p in parts}
    return dict(schema=REVISION, source_sha256=hashlib.sha256(Path(npz_path).read_bytes()).hexdigest(),
                status="配合样件 / 尚未制造放行", stl_units="mm", glb_units="m",
                required_morphology={"honeycomb": "support_iron" in names, "helix": "coolant" in names},
                morphology_note="检查特征体存在；孔形、壁厚、密封与连通仍需几何验证。",
                parts=parts, missing_interfaces=["轴与轴承采购规格", "绕线骨架和装入路径", "端盖与外壳紧固配合", "流道密封接缝", "实测材料参数"],
                assembly_steps=["确认轴、轴承、磁钢及磁芯尺寸", "先打印端盖/支撑配合样件，校准尺寸", "设计绝缘骨架并绕制、检查绕组", "固定磁芯与绕组，检查出线", "装转子、轴承及端盖，检查气隙与手动转动", "完成电气和机械验收后再进行通电试验"])


def package(npz_path):
    meshes = list(part_meshes(npz_path))
    report = manifest(npz_path, meshes)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(report, ensure_ascii=False, indent=2))
        buf = io.StringIO(); writer = csv.writer(buf)
        writer.writerow(["part_id", "零件", "制造路线", "尺寸 mm", "连通体数", "状态"])
        for p in report["parts"]:
            writer.writerow([p["id"],p["label"],p["process"],p["size_mm"],p["components"],report["status"]])
        z.writestr("parts.csv", "\ufeff"+buf.getvalue())
        for name, mesh in meshes:
            mm = mesh.copy(); mm.apply_scale(1000)
            folder = "void_reference" if name == "coolant" else "fit_check_parts"
            z.writestr(f"{folder}/{name}_mm.stl", mm.export(file_type="stl"))
        text = "# MotorGenesis 配合样件包\n\nSTL 单位为毫米；这是原始体素几何，未平滑，也未制造放行。\n\n"
        text += "蜂窝与螺旋是设计约束，不能靠透明度或装饰纹理替代。\n\n"
        for p in report["parts"]:
            text += f"## {p['label']} — {p['process']}\n{p['note']}\n\n"
        text += "## 装配顺序\n" + "\n".join(f"{i+1}. {v}" for i,v in enumerate(report["assembly_steps"]))
        text += "\n\n树脂+四氧化三铁为待测复合材料，不等价于当前铁芯。先测 B-H、损耗、热导率和温度极限，再建立材料卡重新求解。\n"
        z.writestr("制造说明.md", text)
    return out.getvalue()


def casting_molds(npz_path, sample=False):
    """Two tool halves from the ACTUAL stator density, with sprue and bolt bores.

    Uniform cavity offset is a fit-test allowance, not a resin shrinkage model.
    Parting is axial; undercuts and release must be checked before pouring.
    """
    from scipy.ndimage import distance_transform_edt
    from skimage.measure import marching_cubes
    from organic_motor.web.builder import _load_volume
    vol = _load_volume(npz_path)
    with np.load(npz_path, allow_pickle=False) as data:
        solid = vol.iron.copy()
        for k in ("endcap_mask", "rotor_mask", "support_mask", "housing_mask"):
            if k in data.files:
                solid *= 1-np.clip(data[k],0,1)
    occupied = solid > .35
    spacing = np.asarray(vol.spacing)
    base_origin = np.asarray(vol.origin)
    if sample:
        # A small material coupon precedes a full motor casting.
        spacing = np.full(3, .0004)
        base_origin = np.array([-.024,-.024,-.006])
        ax = [base_origin[i]+np.arange(n)*spacing[i] for i,n in enumerate((121,121,31))]
        sx,sy,sz=np.meshgrid(*ax,indexing="ij")
        occupied=(np.hypot(sx,sy)>=.012)&(np.hypot(sx,sy)<=.020)&(np.abs(sz)<=.004)
    if not occupied.any():
        raise ValueError("没有可识别的定子磁芯")
    locations = np.argwhere(occupied)
    lo,hi = locations.min(0),locations.max(0)+1
    pad = np.ceil(.008 / spacing).astype(int)
    occupied = np.pad(occupied[tuple(slice(a,b) for a,b in zip(lo,hi))], tuple((int(p),int(p)) for p in pad))
    origin = base_origin+(lo-pad)*spacing
    axes = [origin[i]+np.arange(occupied.shape[i])*spacing[i] for i in range(3)]
    X,Y,Z = np.meshgrid(*axes,indexing="ij")
    cavity = distance_transform_edt(~occupied,sampling=spacing)-distance_transform_edt(occupied,sampling=spacing)-.0003
    pts = np.argwhere(occupied)
    # A real occupied point guarantees the feed joins the cavity.
    feed = pts[np.argmax(pts[:,2])]
    fx,fy,fz = [axes[i][feed[i]] for i in range(3)]
    sprue = np.maximum(np.hypot(X-fx,Y-fy)-.0025, fz-Z)
    cavity = np.minimum(cavity,sprue)
    for ix in (3,-4):
        for iy in (3,-4):
            cavity = np.minimum(cavity,np.hypot(X-axes[0][ix],Y-axes[1][iy])-.0018)
    split = axes[2][len(axes[2])//2] + spacing[2]*0.37
    box = np.maximum.reduce([axes[0][1]+spacing[0]*.21-X,X-axes[0][-2]+spacing[0]*.21,axes[1][1]+spacing[1]*.21-Y,Y-axes[1][-2]+spacing[1]*.21,axes[2][1]+spacing[2]*.21-Z,Z-axes[2][-2]+spacing[2]*.21])
    result = []
    for name, plane in (("mold_lower", Z-split),("mold_upper",split-Z)):
        field = np.maximum(np.maximum(box,-cavity),plane).astype(np.float32)
        vertices, faces, _, _ = marching_cubes(field,0,spacing=spacing,allow_degenerate=False)
        mesh = trimesh.Trimesh(vertices=vertices+origin,faces=faces,process=True)
        # The closed implicit tool can acquire a tiny triangular hole when
        # marching-cubes removes a zero-area face. Only fill 3/4-edge holes;
        # never ship an open mold or repair a large missing surface silently.
        if not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)
        mesh.fix_normals()
        if not mesh.is_watertight:
            raise ValueError("分模网格未闭合，拒绝导出；需检查型腔薄壁与倒扣")
        result.append((name,mesh))
    return result


def mold_package(npz_path):
    out=io.BytesIO()
    with zipfile.ZipFile(out,"w",zipfile.ZIP_DEFLATED) as z:
        for name,m in casting_molds(npz_path):
            m.apply_scale(1000)
            z.writestr(name+"_mm.stl",m.export(file_type="stl"))
        for name,m in casting_molds(npz_path, sample=True):
            m.apply_scale(1000)
            z.writestr("material_coupon/"+name+"_mm.stl", m.export(file_type="stl"))
        z.writestr("浇注前必读.md", """# FDM 磁芯分模 · 试制工具

根目录为当前 checkpoint 的定子分模。
material_coupon/ 为独立材料试环模具：目标外径 40、内径 24、高 8 mm（另含 0.3 mm 型腔试配余量）。
建议先打印试环模具、测实际浇注试环尺寸和材料性能。试环不是电机的一部分。
STL 单位 mm。上下轴向分模，四个 3.6 mm 配合样孔，上模带 5 mm 浇口。
型腔外扩 0.3 mm 仅为配合试样余量，不是已经测得的固化收缩补偿。

1. 在切片器检查尺寸、壁厚、支撑与分模面；优先先打印局部配合样条。
2. 检查齿部倒扣和脱模方向；目前没有自动脱模认证。倒扣部位需要拆芯或柔性内模。
3. 打印前确认实际打印机尺寸；系统没有假定 A2L 的规格。
4. 模具表面、脱模剂与树脂相容性必须先用小样验证，遵循树脂与粉末供应商的操作说明。
5. 先做复合材料试样，记录粉末质量分数、密度、固化制度和孔隙。
6. 测量 B-H、损耗、温度极限及热导率，建立新材料卡；当前铁芯性能不适用于此配方。
7. 当前模具未完成排气、密封、收缩、磁性能及模具寿命验证，不是批量制造文件。

磁芯模具、蜂窝支撑和流道是不同零件。不要把铜线、永磁体或流体空腔一并浇注。
""")
    return out.getvalue()
