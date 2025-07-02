from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.StlAPI import StlAPI_Reader
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeShell
from OCC.Core.IFSelect import IFSelect_RetDone

# 使用 STL 阅读器来读入 mesh（OBJ需转STL或先用Mesh工具加载）
from OCC.Extend.DataExchange import read_stl_file

# 读取 STL 或通过其他方式加载 mesh 生成 TopoDS_Shape
shape: TopoDS_Shape = read_stl_file("drill.stl")  # 实际上只支持 STL，可以先用 Meshio 转换

# STEP 写出
step_writer = STEPControl_Writer()
step_writer.Transfer(shape, STEPControl_AsIs)
status = step_writer.Write("output_model.step")

if status == IFSelect_RetDone:
    print("STEP 文件导出成功")
else:
    print("STEP 文件导出失败")
