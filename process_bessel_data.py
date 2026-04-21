import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
import os
import re


def read_data_file(txt_file):
    with open(txt_file, "r") as f:
        lines = f.readlines()
    
    data_lines = [line.strip() for line in lines[2:] if line.strip()]
    
    result = []
    for line in data_lines:
        parts = line.split()
        result.append(parts)
    
    return result


def circular_matrix(result, n, cir_status="left", matrix_type="mag"):
    matrix = np.zeros((n, n))
    for i in range(n):
        for k in range(n):
            re_Ex = float(result[i * n + k][3])
            im_Ex = float(result[i * n + k][4])
            re_Ey = float(result[i * n + k][5])
            im_Ey = float(result[i * n + k][6])

            left_E = (
                complex(re_Ex, im_Ex) - complex(0, 1) * complex(re_Ey, im_Ey)
            ) / np.sqrt(2)
            right_E = (
                complex(re_Ex, im_Ex) + complex(0, 1) * complex(re_Ey, im_Ey)
            ) / np.sqrt(2)

            if matrix_type == "mag":
                matrix[i][k] = (
                    np.abs(left_E) if cir_status == "left" else np.abs(right_E)
                )
            elif matrix_type == "phase":
                phase = np.angle(left_E) if cir_status == "left" else np.angle(right_E)
                while phase < 0:
                    phase += 2 * np.pi
                matrix[i][k] = phase
    return matrix


def plot_heat(matrix, x, y, title="plot", plot_type="mag", save_path=None):
    matrix_plot = np.flipud(matrix)
    matrix2 = pd.DataFrame(matrix_plot, index=reversed(y), columns=x)
    sns.set()
    fig, ax = plt.subplots()

    font = {"family": "serif", "color": "darkred", "weight": "normal", "size": 20}

    if plot_type == "phase":
        b = sns.heatmap(matrix2, vmax=2 * np.pi, vmin=0, cbar=False, cmap="jet")
        cbar = b.figure.colorbar(b.collections[0])
        cbar.set_label("Phase/rad", fontdict=font)
        cbar.set_ticks(np.linspace(0, 2 * np.pi, 5))
        cbar.set_ticklabels(
            ("0", chr(960) + "/2", chr(960), "3" + chr(960) + "/2", "2" + chr(960))
        )
        cbar.ax.tick_params(labelsize=20)
    elif plot_type == "mag":
        c = sns.heatmap(matrix2, cmap="jet", cbar=False)
        cbar = c.figure.colorbar(c.collections[0])
        cbar.ax.tick_params(labelsize=20)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_title(title, fontsize=24)
    ax.set_aspect(1)

    x_range = round(x[-1] - x[0], 2)
    y_range = round(y[-1] - y[0], 2)
    plt.xlabel(f"{x_range}", fontdict=font)
    plt.ylabel(f"{y_range}", fontdict=font)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图片已保存: {save_path}")

    plt.close(fig)


def process_single_z_file(txt_file, output_dir):
    txt_filename = os.path.basename(txt_file)
    data_name = os.path.splitext(txt_filename)[0]
    
    z_match = re.search(r'z=(\d+)', data_name)
    z_value = int(z_match.group(1)) if z_match else 0
    
    z_output_dir = os.path.join(output_dir, f"z={z_value}mm")
    if not os.path.exists(z_output_dir):
        os.makedirs(z_output_dir)
    
    result = read_data_file(txt_file)
    
    n = 1
    for i in range(1, len(result)):
        if result[i][1] == result[0][1]:
            n += 1
        else:
            break
    
    total = len(result)
    print(f"处理 {txt_filename}: 总数据点={total}, 网格大小={n}x{n}")
    
    x = np.array([float(result[i][0]) for i in range(n)])
    y = np.array([float(result[i * n][1]) for i in range(n)])
    
    print(f"x范围: {x[0]}~{x[-1]} mm")
    print(f"y范围: {y[0]}~{y[-1]} mm")
    
    matrix_mag_left = circular_matrix(result, n, cir_status="left", matrix_type="mag")
    matrix_phase_left = circular_matrix(result, n, cir_status="left", matrix_type="phase")
    matrix_mag_right = circular_matrix(result, n, cir_status="right", matrix_type="mag")
    matrix_phase_right = circular_matrix(result, n, cir_status="right", matrix_type="phase")
    
    data_files = {
        "Left_circular_Magnitude": matrix_mag_left,
        "Left_circular_Phase": matrix_phase_left,
        "Right_circular_Magnitude": matrix_mag_right,
        "Right_circular_Phase": matrix_phase_right,
    }
    
    for name, matrix in data_files.items():
        df_mm = pd.DataFrame(
            matrix,
            index=[f"{yi:.2f}mm" for yi in y],
            columns=[f"{xi:.2f}mm" for xi in x],
        )
        df_mm.to_csv(os.path.join(z_output_dir, f"{name}_mm.csv"))
        print(f"已保存: {name}_mm.csv")
        
        df_nopos = pd.DataFrame(matrix)
        df_nopos.to_csv(os.path.join(z_output_dir, f"{name}_nopos.csv"))
        print(f"已保存: {name}_nopos.csv")
    
    plot_heat(
        matrix_mag_left,
        x,
        y,
        title=f"Left-circular Magnitude (z={z_value}mm)",
        plot_type="mag",
        save_path=os.path.join(z_output_dir, "Left_circular_Magnitude.jpg"),
    )
    plot_heat(
        matrix_phase_left,
        x,
        y,
        title=f"Left-circular Phase (z={z_value}mm)",
        plot_type="phase",
        save_path=os.path.join(z_output_dir, "Left_circular_Phase.jpg"),
    )
    plot_heat(
        matrix_mag_right,
        x,
        y,
        title=f"Right-circular Magnitude (z={z_value}mm)",
        plot_type="mag",
        save_path=os.path.join(z_output_dir, "Right_circular_Magnitude.jpg"),
    )
    plot_heat(
        matrix_phase_right,
        x,
        y,
        title=f"Right-circular Phase (z={z_value}mm)",
        plot_type="phase",
        save_path=os.path.join(z_output_dir, "Right_circular_Phase.jpg"),
    )
    
    print(f"z={z_value}mm 处理完成！文件保存在: {z_output_dir}/\n")
    return z_output_dir


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    data_dir = "data/20260420_193109_result_Bessel_only_60deg"
    
    output_dir = os.path.join(data_dir, "processed_output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")
    
    z_files = [
        os.path.join(data_dir, "20260420_193109_result_Bessel_only_60deg_z=0.txt"),
        os.path.join(data_dir, "20260420_193109_result_Bessel_only_60deg_z=20.txt"),
        os.path.join(data_dir, "20260420_193109_result_Bessel_only_60deg_z=46.txt"),
        os.path.join(data_dir, "20260420_193109_result_Bessel_only_60deg_z=100.txt"),
    ]
    
    txt_files = [f for f in z_files if os.path.exists(f)]
    
    print(f"找到 {len(txt_files)} 个z平面数据文件:")
    for f in txt_files:
        print(f"  - {os.path.basename(f)}")
    print()
    
    for txt_file in txt_files:
        process_single_z_file(txt_file, output_dir)
    
    print("=" * 50)
    print("所有z平面处理完成！")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
