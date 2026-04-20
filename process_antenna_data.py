import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
import os
import re


def detect_unit(txt_file):
    with open(txt_file, "r") as f:
        header = f.readline()
    if "mm]" in header.lower():
        return "mm"
    elif "in]" in header.lower():
        return "in"
    return "in"


def dh_list(file):
    with open(file, "r") as f:
        f.seek(0)
        result = []
        for eachline in f.readlines()[2:]:
            if not eachline.isspace():
                result.append(eachline.split())
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

    plt.show()


def process_antenna_data(txt_file):
    txt_filename = os.path.basename(txt_file)
    data_name = os.path.splitext(txt_filename)[0]
    output_dir = data_name + "_data"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建文件夹: {output_dir}")

    unit = detect_unit(txt_file)
    print(f"检测到单位: {unit}")

    result_all = dh_list(txt_file)

    n = 1
    for i in range(1, len(result_all)):
        if result_all[i][1] == result_all[0][1]:
            n += 1
        else:
            break

    total = len(result_all)
    n_groups = total // (n * n)
    print(f"总数据点: {total}, 网格大小: {n}x{n}, 数据组数: {n_groups}")

    result = result_all[: n * n]

    x = np.array([float(result[i][0]) for i in range(n)])
    y = np.array([float(result[i * n][1]) for i in range(n)])

    if unit == "in":
        x_display = x * 25.4
        y_display = y * 25.4
        print(f"x范围: {x[0]}~{x[-1]} in ({x_display[0]:.1f}~{x_display[-1]:.1f} mm)")
        print(f"y范围: {y[0]}~{y[-1]} in ({y_display[0]:.1f}~{y_display[-1]:.1f} mm)")
    else:
        x_display = x
        y_display = y
        print(f"x范围: {x[0]}~{x[-1]} mm")
        print(f"y范围: {y[0]}~{y[-1]} mm")

    matrix_mag_left = circular_matrix(result, n, cir_status="left", matrix_type="mag")
    matrix_phase_left = circular_matrix(
        result, n, cir_status="left", matrix_type="phase"
    )
    matrix_mag_right = circular_matrix(result, n, cir_status="right", matrix_type="mag")
    matrix_phase_right = circular_matrix(
        result, n, cir_status="right", matrix_type="phase"
    )

    data_files = {
        "Left_circular_Magnitude": matrix_mag_left,
        "Left_circular_Phase": matrix_phase_left,
        "Right_circular_Magnitude": matrix_mag_right,
        "Right_circular_Phase": matrix_phase_right,
    }

    for name, matrix in data_files.items():
        if unit == "in":
            df_mm = pd.DataFrame(
                matrix,
                index=[f"{yi:.2f}mm" for yi in y_display],
                columns=[f"{xi:.2f}mm" for xi in x_display],
            )
            df_mm.to_csv(os.path.join(output_dir, f"{name}_mm.csv"))
            print(f"已保存: {name}_mm.csv")
        else:
            df_mm = pd.DataFrame(
                matrix,
                index=[f"{yi:.2f}mm" for yi in y_display],
                columns=[f"{xi:.2f}mm" for xi in x_display],
            )
            df_mm.to_csv(os.path.join(output_dir, f"{name}_mm.csv"))
            print(f"已保存: {name}_mm.csv")

        df_nopos = pd.DataFrame(matrix)
        df_nopos.to_csv(os.path.join(output_dir, f"{name}_nopos.csv"))
        print(f"已保存: {name}_nopos.csv")

    if unit == "in":
        plot_heat(
            matrix_mag_left,
            x_display,
            y_display,
            title="Left-circular Magnitude",
            plot_type="mag",
            save_path=os.path.join(output_dir, "Left_circular_Magnitude.jpg"),
        )
        plot_heat(
            matrix_phase_left,
            x_display,
            y_display,
            title="Left-circular Phase",
            plot_type="phase",
            save_path=os.path.join(output_dir, "Left_circular_Phase.jpg"),
        )
        plot_heat(
            matrix_mag_right,
            x_display,
            y_display,
            title="Right-circular Magnitude",
            plot_type="mag",
            save_path=os.path.join(output_dir, "Right_circular_Magnitude.jpg"),
        )
        plot_heat(
            matrix_phase_right,
            x_display,
            y_display,
            title="Right-circular Phase",
            plot_type="phase",
            save_path=os.path.join(output_dir, "Right_circular_Phase.jpg"),
        )
    else:
        plot_heat(
            matrix_mag_left,
            x_display,
            y_display,
            title="Left-circular Magnitude",
            plot_type="mag",
            save_path=os.path.join(output_dir, "Left_circular_Magnitude.jpg"),
        )
        plot_heat(
            matrix_phase_left,
            x_display,
            y_display,
            title="Left-circular Phase",
            plot_type="phase",
            save_path=os.path.join(output_dir, "Left_circular_Phase.jpg"),
        )
        plot_heat(
            matrix_mag_right,
            x_display,
            y_display,
            title="Right-circular Magnitude",
            plot_type="mag",
            save_path=os.path.join(output_dir, "Right_circular_Magnitude.jpg"),
        )
        plot_heat(
            matrix_phase_right,
            x_display,
            y_display,
            title="Right-circular Phase",
            plot_type="phase",
            save_path=os.path.join(output_dir, "Right_circular_Phase.jpg"),
        )

    print(f"\n处理完成！所有文件保存在: {output_dir}/")
    return output_dir


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    txt_files = [
        # "20260416_hex_180_metasurface_center_colli_z-5.txt",
        # "20260416_hex_180_metasurface_center_colli_z100.txt",
        "20260416_hex_360_metasurface_center_colli_z150.txt"
    ]
    for txt_file in txt_files:
        if os.path.exists(txt_file):
            process_antenna_data(txt_file)
        else:
            print(f"文件不存在: {txt_file}")
