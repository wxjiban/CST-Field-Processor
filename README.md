# CST-Field-Processor / CST电磁场数据处理工具

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

CST仿真电磁场数据处理与分析工具集，支持近场/远场数据读取、圆极化分解、3D空间可视化等功能。

A comprehensive toolkit for processing and analyzing CST electromagnetic field simulation data, supporting near-field/far-field data reading, circular polarization decomposition, and 3D spatial visualization.

---

## 功能特性 / Features

- **圆极化分解**: 从线极化电场(Ex, Ey)计算左旋/右旋圆极化分量
- **近场截面分析**: 提取XOZ/YOZ/XOY截面数据并生成热力图
- **Z轴中心线提取**: 沿传播方向提取一维幅度/相位变化曲线
- **远场UV投影**: 球坐标→直角坐标投影，避免极坐标螺旋失真
- **多格式输出**: PNG高清图 + CSV数据文件

---

## 脚本说明 / Scripts

### 1. `process_circular_polarization_3d.py` (主脚本)

功能最全面的核心处理脚本。

**功能模块**:
| 函数 | 功能 |
|------|------|
| `process_cross_section()` | 提取XOZ/YOZ截面数据，生成圆极化幅度热力图 |
| `process_z_axis_center_line()` | 提取Z轴中心线(x=0,y=0)的1D幅度/相位数据 |
| `plot_farfield_ortho()` | 远场正交投影UV图（球坐标→直角坐标，避免螺旋失真） |

**输入数据格式**:
```
x [mm]   y [mm]   z [mm]   ExRe   ExIm   EyRe   EyIm   EzRe   EzIm
```

**输出目录结构**:
```
processed_output/
├── z_axis/                    # Z轴中心线结果
│   ├── z_axis_center_line.csv # 1D数据(z, Ex, Ey, 幅度, 相位)
│   ├── z_axis_magnitude.png   # 幅度随Z变化曲线
│   └── z_axis_phase.png       # 相位随Z变化曲线
└── ortho/                     # 远场正交投影结果
    ├── farfield_left_uv.png   # 左旋UV平面投影
    ├── farfield_right_uv.png  # 右旋UV平面投影
    └── farfield_ortho_combined.png # 左右旋对比图
```

### 2. `process_antenna_data.py`

天线近场数据处理脚本，适用于规则网格数据。

**功能**:
- 读取CST导出的天线近场电场数据
- 构建N×N矩阵，计算左旋/右旋圆极化分量
- 生成幅度热力图和相位分布图
- 支持毫米(mm)/英寸(in)单位自动检测

### 3. `process_bessel_data.py`

贝塞尔波束专用数据处理脚本。

**功能**:
- 处理贝塞尔波束仿真数据
- 圆极化分解与可视化
- 适用于理论场与仿真场对比分析

---

## 技术细节 / Technical Details

### 圆极化分解公式

```python
E_left  = (Ex - j·Ey) / √2    # 左旋圆极化
E_right = (Ex + j·Ey) / √2    # 右旋圆极化
```

### 远场UV投影（避免螺旋失真）

```python
# 球坐标 → 直角坐标投影
U = sin(θ)·cos(φ)
V = sin(θ)·sin(φ)

# 插值到规则网格
DD = griddata((u_flat, v_flat), d_flat, (UU, VV), method='linear')

# 圆形mask过滤
outside = UU**2 + VV**2 > 1.0
DD[outside] = np.nan
```

### 数据排列方向

CST输出格式：Theta内层快变，Phi外层慢变
```python
data_2d = data.reshape((n_phi, n_theta))  # 行=phi, 列=theta
```

---

## 依赖 / Dependencies

```
numpy
matplotlib
pandas
seaborn
scipy
```

安装:
```bash
pip install numpy matplotlib pandas seaborn scipy
```

---

## 使用方法 / Usage

```bash
# 运行主脚本
python process_circular_polarization_3d.py

# 运行天线数据处理
python process_antenna_data.py

# 运行贝塞尔数据处理
python process_bessel_data.py
```

在脚本中修改输入文件路径和输出目录即可处理自己的数据。

---

## 项目结构 / Project Structure

```
CST-Field-Processor/
├── process_circular_polarization_3d.py  # 主处理脚本
├── process_antenna_data.py              # 天线数据处理
├── process_bessel_data.py               # 贝塞尔数据处理
├── data/                                # 数据目录
│   └── 20260420_193109_result_Bessel_only_60deg/
│       ├── processed_output/            # 输出结果
│       │   ├── z_axis/                  # Z轴中心线结果
│       │   └── ortho/                   # 远场投影结果
│       └── *.txt                        # 输入数据文件
├── .gitignore
└── README.md
```

---

## 许可证 / License

MIT License
