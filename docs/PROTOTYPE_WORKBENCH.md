# 原型零件工作台

本轮将材料视图扩展为 artifact 驱动的零件/功能组视图。实际几何仍来自当前 checkpoint，不生成假装配图片。

## 使用

```sh
.venv/bin/python -m organic_motor.reports.assembly_demo --full
.venv/bin/python -m organic_motor.web --out organic_motor/out --host 127.0.0.1 --port 8011
```

选择 assembly。点击零件名称隔离，勾选框控制显示；爆炸滑块逐件展开。查看蜂窝/螺旋按钮展开两个功能组。恢复装配恢复完整几何。

下载零件包获得独立毫米 STL、零件 CSV、源文件 SHA256、尺寸/闭合性/连通体记录及制造说明。蓝色流道另存 void_reference，不能作为实体打印。导出使用固定 0.35 等值面且不平滑，独立于显示爆炸位置。

分模包包含当前定子上下模，以及独立的材料试环上下模（目标外径 40、内径 24、高 8 mm，型腔另有 0.3 mm 试配余量）。上模有浇口；四个孔供试配夹紧。所有模具均为试制工具，尚未验证脱模、排气、密封及固化收缩。

## 用户制造路线

- 设备：用户有拓竹 A1 和其称为 A2L 的 FDM 打印机；未擅自假定第二台的成型尺寸。
- 目标：FDM 打印模具，浇注树脂/四氧化三铁磁芯。
- 铜绕组使用漆包铜线，磁钢采购；绝缘骨架、轴和轴承接口仍需确定。
- 当前电磁铁材料没有改成复合材料：未知材料不能伪造 B-H、损耗或热参数。应先测实际配方试样。

工业 SMC 并非任意树脂粉末混合物。参考：[Höganäs SMC](https://www.hoganas.com/en/powder-technologies/soft-magnetic-composites/)。打印配合应先验证方向和公差：[Prusa 建模指南](https://help.prusa3d.com/article/modeling-with-3d-printing-in-mind_164135?product=core-one)。

## 设计约束

OrganicPrototypeSpec 在 assembly --full 生成时强制保留蜂窝和至少一圈螺旋，拒绝空特征、壁厚不小于单元尺寸、以及壁厚不足两个体素的候选。装配几何提升至 128×128×96。这个门槛只检查最低表达能力，并非制造或力学认证；未自动接入所有旧优化入口。

## 后续工程缺口

1. 将功能组继续变成有连接接口的独立制造零件，尤其磁芯分段和绕线骨架。
2. 分件来自明确掩码，但旧体素模型仍有碎片/材料边界问题。检查连通体记录，不把闭合 STL 等同可制造。
3. 不同零件的螺栓孔、轴承座、定位止口必须成对验证；EndCapGenerator 的 bearing_od 已修正为直径（默认 28 mm 保留原有孔径），并修复止口朝内凸出；实际轴承规格仍需确定。
4. 固化收缩、倒扣、排气与模具表面处理需要试验；0.3 mm 只是型腔试配余量。
5. 复合材料卡完成后重新求解电磁与热问题；本轮没有重新认证电机性能。
6. 旋转件当前仍主要按名称前缀分组，轴与轴承要分别建模；不能把静止轴承外圈随转子旋转。

## 验证

```sh
.venv/bin/python -m pytest tests/test_part_workbench.py tests/test_model_artifact.py tests/test_slice_provenance.py -q
```

覆盖独立端盖、GLB 节点、STL 米到毫米转换、上下模闭合、试环模具、缺失物理场接口及强制形态约束。浏览器实际检查零件隔离、蜂窝/螺旋视图与缺失温度场提示。
