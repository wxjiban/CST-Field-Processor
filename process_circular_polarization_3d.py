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


def get_unique_z_values(result_all):
    z_values = []
    seen = set()
    for row in result_all:
        z = float(row[2])
        if z not in seen:
            seen.add(z)
            z_values.append(z)
    return sorted(z_values)


def get_unique_y_values(result_all):
    y_values = []
    seen = set()
    for row in result_all:
        y = float(row[1])
        if y not in seen:
            seen.add(y)
            y_values.append(y)
    return sorted(y_values)


def get_unique_x_values(result_all):
    x_values = []
    seen = set()
    for row in result_all:
        x = float(row[0])
        if x not in seen:
            seen.add(x)
            x_values.append(x)
    return sorted(x_values)


def find_nearest_value(target, values):
    arr = np.array(values)
    idx = np.argmin(np.abs(arr - target))
    return arr[idx], idx


def plot_heat(matrix, x, y, title="plot", plot_type="mag", save_path=None, vmax=None):
    matrix_plot = np.flipud(matrix.transpose())
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
        if vmax is None:
            vmax = 1
        c = sns.heatmap(matrix2, vmax=vmax, cmap="jet", cbar=False)
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

    plt.close()


def plot_farfield_ortho(txt_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = np.loadtxt(txt_file, skiprows=2)

    theta = data[:, 0]
    phi   = data[:, 1]
    abs_left    = data[:, 3]
    abs_right   = data[:, 5]

    theta_unique = np.unique(theta)
    phi_unique   = np.unique(phi)
    n_theta = len(theta_unique)
    n_phi   = len(phi_unique)

    print(f"Theta: {n_theta}点, {theta_unique.min():.1f}°~{theta_unique.max():.1f}°")
    print(f"Phi:   {n_phi}点, {phi_unique.min():.1f}°~{phi_unique.max():.1f}°")
    print(f"前5行:\n{data[:5, :2]}")

    # ✅ 你的数据：theta快变，phi慢变
    # → reshape成 (n_phi, n_theta)，行=phi，列=theta
    left_2d  = abs_left.reshape((n_phi, n_theta))
    right_2d = abs_right.reshape((n_phi, n_theta))

    # ✅ 构造UV网格，与reshape方向一致
    # THETA shape: (n_phi, n_theta)
    PHI_grid, THETA_grid = np.meshgrid(
        np.deg2rad(phi_unique),
        np.deg2rad(theta_unique),
        indexing='ij'          # ij: 第0轴=phi, 第1轴=theta → 匹配reshape
    )

    # 只保留 theta <= 90° 的半球（前向散射）
    mask_theta = theta_unique <= 90.0
    THETA_plot = THETA_grid[:, mask_theta]
    PHI_plot   = PHI_grid[:, mask_theta]
    left_plot  = left_2d[:, mask_theta]
    right_plot = right_2d[:, mask_theta]

    U = np.sin(THETA_plot) * np.cos(PHI_plot)
    V = np.sin(THETA_plot) * np.sin(PHI_plot)

    font = {"family": "serif", "color": "darkred", "weight": "normal", "size": 16}

    def plot_uv(data_2d, title, cbar_label, save_path):
        fig, ax = plt.subplots(figsize=(8, 8))

        # 散点插值到规则网格再画，避免contourf三角剖分乱序问题
        from scipy.interpolate import griddata
        u_flat = U.ravel()
        v_flat = V.ravel()
        d_flat = data_2d.ravel()

        u_lin = np.linspace(-1, 1, 500)
        v_lin = np.linspace(-1, 1, 500)
        UU, VV = np.meshgrid(u_lin, v_lin)

        DD = griddata((u_flat, v_flat), d_flat, (UU, VV), method='linear')

        # 圆形mask（UV单位圆外设为nan）
        outside = UU**2 + VV**2 > 1.0
        DD[outside] = np.nan

        im = ax.pcolormesh(UU, VV, DD, cmap='jet', shading='auto')

        # 画参考圆：θ=30°, 60°, 90°
        for theta_ref in [30, 60, 90]:
            r = np.sin(np.deg2rad(theta_ref))
            circle = plt.Circle((0, 0), r, color='white',
                                 fill=False, linewidth=0.8, linestyle='--', alpha=0.6)
            ax.add_patch(circle)
            ax.text(0, r, f'{theta_ref}°', color='white',
                    fontsize=9, ha='center', va='bottom')

        ax.set_aspect('equal')
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel('u = sin(θ)cos(φ)', fontdict=font)
        ax.set_ylabel('v = sin(θ)sin(φ)', fontdict=font)
        cbar = plt.colorbar(im, ax=ax, pad=0.05, shrink=0.85)
        cbar.set_label(cbar_label, fontdict=font)
        ax.set_title(title, fontdict=font, pad=15)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {save_path}")
        plt.close()

    plot_uv(left_plot,  'Far-field LCP Magnitude', 'Abs(Left) [dB(m²)]',
            os.path.join(output_dir, "farfield_left_uv.png"))

    plot_uv(right_plot, 'Far-field RCP Magnitude', 'Abs(Right) [dB(m²)]',
            os.path.join(output_dir, "farfield_right_uv.png"))
    
    # Save combined figure
    combined_path = os.path.join(output_dir, "farfield_ortho_combined.png")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    from scipy.interpolate import griddata
    for ax, data_2d, title in zip(axes, [left_plot, right_plot], ['LCP', 'RCP']):
        u_flat = U.ravel()
        v_flat = V.ravel()
        d_flat = data_2d.ravel()
        u_lin = np.linspace(-1, 1, 500)
        v_lin = np.linspace(-1, 1, 500)
        UU, VV = np.meshgrid(u_lin, v_lin)
        DD = griddata((u_flat, v_flat), d_flat, (UU, VV), method='linear')
        outside = UU**2 + VV**2 > 1.0
        DD[outside] = np.nan
        im = ax.pcolormesh(UU, VV, DD, cmap='jet', shading='auto')
        for theta_ref in [30, 60, 90]:
            r = np.sin(np.deg2rad(theta_ref))
            circle = plt.Circle((0, 0), r, color='white', fill=False, linewidth=0.8, linestyle='--', alpha=0.6)
            ax.add_patch(circle)
        ax.set_aspect('equal')
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(title, fontdict=font)
        plt.colorbar(im, ax=ax, pad=0.05, shrink=0.85)
    
    plt.tight_layout()
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"已保存: {combined_path}")
    plt.close()
    
def process_cross_section(txt_file, z_coords=None, y_fixed=0.0, x_fixed=0.0):
    txt_filename = os.path.basename(txt_file)
    data_name = os.path.splitext(txt_filename)[0]
    output_dir = data_name + "_cross_data"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建文件夹: {output_dir}")

    unit = detect_unit(txt_file)
    print(f"检测到单位: {unit}")

    result_all = dh_list(txt_file)

    z_values_in_data = get_unique_z_values(result_all)
    y_values_in_data = get_unique_y_values(result_all)
    x_values_in_data = get_unique_x_values(result_all)

    n_x = len(x_values_in_data)
    n_y = len(y_values_in_data)
    n_z = len(z_values_in_data)

    print(f"x方向点数: {n_x}")
    print(f"y方向点数: {n_y}")
    print(f"z方向点数: {n_z}")

    y_fixed_matched, y_idx = find_nearest_value(y_fixed, y_values_in_data)
    x_fixed_matched, x_idx = find_nearest_value(x_fixed, x_values_in_data)
    print(f"XOZ面: y={y_fixed_matched} (数据中最接近{y_fixed}的值)")
    print(f"YOZ面: x={x_fixed_matched} (数据中最接近{x_fixed}的值)")

    xoz_output_dir = os.path.join(output_dir, "XOZ")
    yoz_output_dir = os.path.join(output_dir, "YOZ")
    if not os.path.exists(xoz_output_dir):
        os.makedirs(xoz_output_dir)
    if not os.path.exists(yoz_output_dir):
        os.makedirs(yoz_output_dir)

    xoz_data = []
    yoz_data = []

    for row in result_all:
        y_val = float(row[1])
        x_val = float(row[0])

        if abs(y_val - y_fixed_matched) < 1e-6:
            xoz_data.append(row)
        if abs(x_val - x_fixed_matched) < 1e-6:
            yoz_data.append(row)

    expected_xoz = n_x * n_z
    expected_yoz = n_y * n_z

    if len(xoz_data) != expected_xoz:
        print(f"警告: XOZ面数据点数({len(xoz_data)})不等于{expected_xoz}，跳过XOZ")
        xoz_data = []
    if len(yoz_data) != expected_yoz:
        print(f"警告: YOZ面数据点数({len(yoz_data)})不等于{expected_yoz}，跳过YOZ")
        yoz_data = []

    if xoz_data:
        x_xoz = np.array([float(row[0]) for row in xoz_data])
        z_xoz = np.array([float(row[2]) for row in xoz_data])

        if unit == "in":
            x_display = x_xoz * 25.4
            z_display = z_xoz * 25.4
        else:
            x_display = x_xoz
            z_display = z_xoz

        x_display_unique = np.unique(x_display)
        z_display_unique = np.unique(z_display)

        xoz_sorted_idx = np.lexsort((x_display, z_display))
        xoz_data_sorted = [xoz_data[i] for i in xoz_sorted_idx]

        matrix_mag_left = np.zeros((n_z, n_x))
        matrix_mag_right = np.zeros((n_z, n_x))

        for zi in range(n_z):
            for xi in range(n_x):
                idx = zi * n_x + xi
                row = xoz_data_sorted[idx]
                re_Ex = float(row[3])
                im_Ex = float(row[4])
                re_Ey = float(row[5])
                im_Ey = float(row[6])

                left_E = (complex(re_Ex, im_Ex) - complex(0, 1) * complex(re_Ey, im_Ey)) / np.sqrt(2)
                right_E = (complex(re_Ex, im_Ex) + complex(0, 1) * complex(re_Ey, im_Ey)) / np.sqrt(2)

                matrix_mag_left[zi][xi] = np.abs(left_E)
                matrix_mag_right[zi][xi] = np.abs(right_E)

        plot_heat(
            matrix_mag_left,
            z_display_unique,
            x_display_unique,
            title=f"XOZ Left-circular Magnitude (y={y_fixed_matched:.4f}mm)",
            plot_type="mag",
            save_path=os.path.join(xoz_output_dir, "Left_circular_Magnitude.jpg"),
        )
        plot_heat(
            matrix_mag_right,
            z_display_unique,
            x_display_unique,
            title=f"XOZ Right-circular Magnitude (y={y_fixed_matched:.4f}mm)",
            plot_type="mag",
            save_path=os.path.join(xoz_output_dir, "Right_circular_Magnitude.jpg"),
        )

        df_left = pd.DataFrame(matrix_mag_left)
        df_left.to_csv(os.path.join(xoz_output_dir, "Left_circular_Magnitude_nopos.csv"), index=False, header=False)
        df_right = pd.DataFrame(matrix_mag_right)
        df_right.to_csv(os.path.join(xoz_output_dir, "Right_circular_Magnitude_nopos.csv"), index=False, header=False)
        print(f"XOZ面数据已保存到: {xoz_output_dir}/")

    if yoz_data:
        y_yoz = np.array([float(row[1]) for row in yoz_data])
        z_yoz = np.array([float(row[2]) for row in yoz_data])

        if unit == "in":
            y_display = y_yoz * 25.4
            z_display = z_yoz * 25.4
        else:
            y_display = y_yoz
            z_display = z_yoz

        y_display_unique = np.unique(y_display)
        z_display_unique = np.unique(z_display)

        yoz_sorted_idx = np.lexsort((y_display, z_display))
        yoz_data_sorted = [yoz_data[i] for i in yoz_sorted_idx]

        matrix_mag_left = np.zeros((n_z, n_y))
        matrix_mag_right = np.zeros((n_z, n_y))

        for zi in range(n_z):
            for yi in range(n_y):
                idx = zi * n_y + yi
                row = yoz_data_sorted[idx]
                re_Ex = float(row[3])
                im_Ex = float(row[4])
                re_Ey = float(row[5])
                im_Ey = float(row[6])

                left_E = (complex(re_Ex, im_Ex) - complex(0, 1) * complex(re_Ey, im_Ey)) / np.sqrt(2)
                right_E = (complex(re_Ex, im_Ex) + complex(0, 1) * complex(re_Ey, im_Ey)) / np.sqrt(2)

                matrix_mag_left[zi][yi] = np.abs(left_E)
                matrix_mag_right[zi][yi] = np.abs(right_E)

        plot_heat(
            matrix_mag_left,
            z_display_unique,
            y_display_unique,
            title=f"YOZ Left-circular Magnitude (x={x_fixed_matched:.4f}mm)",
            plot_type="mag",
            save_path=os.path.join(yoz_output_dir, "Left_circular_Magnitude.jpg"),
        )
        plot_heat(
            matrix_mag_right,
            z_display_unique,
            y_display_unique,
            title=f"YOZ Right-circular Magnitude (x={x_fixed_matched:.4f}mm)",
            plot_type="mag",
            save_path=os.path.join(yoz_output_dir, "Right_circular_Magnitude.jpg"),
        )

        df_left = pd.DataFrame(matrix_mag_left)
        df_left.to_csv(os.path.join(yoz_output_dir, "Left_circular_Magnitude_nopos.csv"), index=False, header=False)
        df_right = pd.DataFrame(matrix_mag_right)
        df_right.to_csv(os.path.join(yoz_output_dir, "Right_circular_Magnitude_nopos.csv"), index=False, header=False)
        print(f"YOZ面数据已保存到: {yoz_output_dir}/")

    print(f"\n处理完成！文件保存在: {output_dir}/")
    return output_dir


def process_z_axis_center_line(txt_file, output_dir, x_target=0.0, y_target=0.0):
    txt_filename = os.path.basename(txt_file)
    data_name = os.path.splitext(txt_filename)[0]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    unit = detect_unit(txt_file)
    print(f"检测到单位: {unit}")

    print("正在读取数据文件...")
    result_all = dh_list(txt_file)
    print(f"总数据量: {len(result_all)} 行")

    z_values = get_unique_z_values(result_all)
    n_z = len(z_values)
    print(f"Z方向点数: {n_z}")
    print(f"Z范围: {z_values[0]:.2f} ~ {z_values[-1]:.2f}")

    x_values_in_data = get_unique_x_values(result_all)
    y_values_in_data = get_unique_y_values(result_all)

    x_matched, _ = find_nearest_value(x_target, x_values_in_data)
    y_matched, _ = find_nearest_value(y_target, y_values_in_data)
    print(f"提取中心线: x={x_matched:.4f}, y={y_matched:.4f}")

    tolerance = 1e-6
    center_data = []
    for row in result_all:
        x_val = float(row[0])
        y_val = float(row[1])
        if abs(x_val - x_matched) < tolerance and abs(y_val - y_matched) < tolerance:
            center_data.append(row)

    print(f"提取到中心线数据: {len(center_data)} 个点")

    if len(center_data) != n_z:
        print(f"警告: 中心线数据点数({len(center_data)})与Z方向点数({n_z})不匹配")

    center_data_sorted = sorted(center_data, key=lambda r: float(r[2]))

    z_axis = []
    mag_left = []
    mag_right = []
    phase_left = []
    phase_right = []
    ex_re_list = []
    ex_im_list = []
    ey_re_list = []
    ey_im_list = []

    for row in center_data_sorted:
        z = float(row[2])
        re_Ex = float(row[3])
        im_Ex = float(row[4])
        re_Ey = float(row[5])
        im_Ey = float(row[6])

        Ex = complex(re_Ex, im_Ex)
        Ey = complex(re_Ey, im_Ey)

        left_E = (Ex - complex(0, 1) * Ey) / np.sqrt(2)
        right_E = (Ex + complex(0, 1) * Ey) / np.sqrt(2)

        z_axis.append(z)
        mag_left.append(np.abs(left_E))
        mag_right.append(np.abs(right_E))
        phase_left.append(np.angle(left_E) % (2 * np.pi))
        phase_right.append(np.angle(right_E) % (2 * np.pi))
        ex_re_list.append(re_Ex)
        ex_im_list.append(im_Ex)
        ey_re_list.append(re_Ey)
        ey_im_list.append(im_Ey)

    z_axis = np.array(z_axis)
    mag_left = np.array(mag_left)
    mag_right = np.array(mag_right)
    phase_left = np.array(phase_left)
    phase_right = np.array(phase_right)

    csv_data = pd.DataFrame({
        'z_mm': z_axis,
        'Ex_re': ex_re_list,
        'Ex_im': ex_im_list,
        'Ey_re': ey_re_list,
        'Ey_im': ey_im_list,
        'mag_left': mag_left,
        'mag_right': mag_right,
        'phase_left': phase_left,
        'phase_right': phase_right
    })
    csv_path = os.path.join(output_dir, "z_axis_center_line.csv")
    csv_data.to_csv(csv_path, index=False)
    print(f"CSV已保存: {csv_path}")

    sns.set()
    fig, ax = plt.subplots(figsize=(10, 6))

    font = {"family": "serif", "color": "darkred", "weight": "normal", "size": 16}

    ax.plot(z_axis, mag_left, 'b-', linewidth=2, label='Left-circular')
    ax.plot(z_axis, mag_right, 'r-', linewidth=2, label='Right-circular')

    ax.set_xlabel('Z [mm]', fontdict=font)
    ax.set_ylabel('Magnitude [V/m]', fontdict=font)
    ax.set_title(f'Z-axis Center Line (x={x_matched:.2f}, y={y_matched:.2f})', fontdict=font)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    mag_save_path = os.path.join(output_dir, "z_axis_magnitude.png")
    plt.savefig(mag_save_path, dpi=300, bbox_inches="tight")
    print(f"幅度图已保存: {mag_save_path}")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(z_axis, phase_left, 'b-', linewidth=2, label='Left-circular phase')
    ax.plot(z_axis, phase_right, 'r-', linewidth=2, label='Right-circular phase')

    ax.set_xlabel('Z [mm]', fontdict=font)
    ax.set_ylabel('Phase [rad]', fontdict=font)
    ax.set_title(f'Z-axis Center Line Phase (x={x_matched:.2f}, y={y_matched:.2f})', fontdict=font)
    ax.legend(fontsize=14)
    ax.set_ylim(0, 2 * np.pi)
    ax.set_yticks(np.linspace(0, 2 * np.pi, 5))
    ax.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    ax.grid(True, alpha=0.3)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    phase_save_path = os.path.join(output_dir, "z_axis_phase.png")
    plt.savefig(phase_save_path, dpi=300, bbox_inches="tight")
    print(f"相位图已保存: {phase_save_path}")
    plt.close()

    print(f"\nZ轴中心线处理完成！文件保存在: {output_dir}/")
    return output_dir


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    bessel_data_dir = "data/20260420_193109_result_Bessel_only_60deg"
    bessel_efield = os.path.join(bessel_data_dir, "20260420_193109_result_Bessel_only_60deg_efield.txt")
    bessel_farfield = os.path.join(bessel_data_dir, "20260420_193109_result_Bessel_only_60deg_farfield_2d_ortho.txt")
    bessel_output_base = os.path.join(bessel_data_dir, "processed_output")
    
    ortho_output_dir = os.path.join(bessel_output_base, "ortho")
    z_axis_output_dir = os.path.join(bessel_output_base, "z_axis")

    # Delete old results
    import shutil
    for d in [ortho_output_dir, z_axis_output_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"已删除旧目录: {d}")

    if os.path.exists(bessel_efield):
        print(f"\n{'='*60}")
        print(f"=== Z轴中心线处理: {bessel_efield} ===")
        print(f"{'='*60}")
        process_z_axis_center_line(
            bessel_efield,
            output_dir=z_axis_output_dir,
            x_target=0.0,
            y_target=0.0
        )
    else:
        print(f"文件不存在: {bessel_efield}")

    if os.path.exists(bessel_farfield):
        print(f"\n{'='*60}")
        print(f"=== 远场极化分布处理: {bessel_farfield} ===")
        print(f"{'='*60}")
        plot_farfield_ortho(
            bessel_farfield,
            output_dir=ortho_output_dir
        )
    else:
        print(f"文件不存在: {bessel_farfield}")
