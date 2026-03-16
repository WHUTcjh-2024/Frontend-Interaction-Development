# 必须放在最开头，确保matplotlib后端生效
import matplotlib
matplotlib.use('Agg')
import math
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import traceback
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)
LAMBDA0 = 632.8 * 0.000000001

def calculate_surface_tension(f, delta_X_cm, H0, h, L, rho, sigma0=None):
    """
    :param f: 激励频率，单位Hz
    :param delta_X_cm: 零级与一级条纹间距，单位cm
    :param H0: 零级亮斑离桌面高度，单位cm
    :param h: 水面高度，单位cm
    :param L: 振动点到光屏的水平距离，单位cm
    :param rho: 待测液体密度，单位kg/m³
    :param sigma0: 表面张力系数标准值（选填），单位N/m
    :return: 计算结果
    """
    if f <= 0:
        raise ValueError("激励频率f必须大于0")
    if delta_X_cm <= 0:
        raise ValueError("条纹间距Δx必须大于0")
    if L <= 0:
        raise ValueError("振动点到光屏的距离L必须大于0")
    if rho <= 0:
        raise ValueError("液体密度ρ必须大于0")
    if H0 <= h:
        raise ValueError("零级亮斑离桌面高度H0必须大于水面高度h")
    if sigma0 is not None and sigma0 <= 0:
        raise ValueError("表面张力系数标准值σ₀必须大于0")

    H0_m = H0 * 0.01
    h_m = h * 0.01
    L_m = L * 0.01
    delta_X_m = delta_X_cm * 0.01
    H = H0_m - h_m

    alpha = math.atan(H / L_m)
    beta = math.atan((H + delta_X_m) / L_m)
    delta = abs(beta - alpha)

    term1 = 2 * math.pi / LAMBDA0
    term2 = math.sin(delta / 2)
    term3 = math.sin(alpha + delta / 2) + math.sin(alpha - delta / 2)
    k = term1 * term2 * term3

    omega = 2 * math.pi * f
    sigma = (rho * omega ** 2) / (k ** 3)

    relative_error = None
    if sigma0 is not None and sigma0 != 0:
        relative_error = (abs(sigma - sigma0) / sigma0) * 100

    return {
        "delta": round(delta, 6),
        "k": round(k, 0),
        "sigma": round(sigma, 4),
        "relative_error": round(relative_error, 3) if relative_error is not None else None
    }

# ===================== 绘图逻辑优化 匹配参考图样式 =====================
def fit_and_plot(experiment_data, rho, sigma0=None):
    """
    强制过原点的最小二乘线性拟合（物理模型不变），绘图优化让数据点占据坐标纸主要部分
    :param experiment_data: 实验数据列表
    :param rho: 液体密度 kg/m³
    :param sigma0: 标准表面张力系数 N/m（选填）
    :return: 拟合参数、base64格式图片
    """
    try:
        # 1. 严格数据清洗与校验
        valid_data = []
        for idx, d in enumerate(experiment_data):
            try:
                k = float(d['k'])
                f = float(d['f'])
                if k <= 0 or f <= 0:
                    continue
                valid_data.append((k, f))
            except Exception as e:
                app.logger.warning(f"第{idx+1}组数据解析失败: {str(e)}")
                continue

        if len(valid_data) < 2:
            raise ValueError(f"有效数据仅{len(valid_data)}组，至少需要2组有效实验数据才能拟合")

        # 2. 物理量计算
        valid_data = np.array(valid_data, dtype=np.float64)
        k_list = valid_data[:, 0]
        f_list = valid_data[:, 1]
        x = k_list ** 3                # x轴：k³  单位 m^-3
        y = (2 * np.pi * f_list) ** 2  # y轴：ω² 单位 (rad/s)²

        # 3. 强制过原点最小二乘拟合
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x ** 2)
        if abs(sum_x2) < 1e-10:
            raise ValueError("波数数据异常，无法拟合")
        a = sum_xy / sum_x2
        if a <= 0:
            raise ValueError("拟合斜率为负，数据不符合物理规律，请检查实验数据")

        # 4. 拟合优度与物理参数计算
        y_pred = a * x
        sum_sq_res = np.sum((y - y_pred) ** 2)
        sum_sq_tot = np.sum(y ** 2)
        r_squared = 1 - (sum_sq_res / sum_sq_tot) if abs(sum_sq_tot) > 1e-10 else 0
        r_squared = np.clip(r_squared, 0, 1)

        sigma_fit = float(a * rho)
        sigma_fit_rounded = round(sigma_fit, 4)
        a_rounded = round(float(a), 8)
        r_squared_rounded = round(float(r_squared), 4)

        relative_error_fit = None
        if sigma0 is not None and sigma0 > 0:
            relative_error_fit = round(float((abs(sigma_fit - sigma0) / sigma0) * 100), 2)

        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        fig, ax = plt.subplots(figsize=(12, 7), dpi=120)

        # 1. 绘制蓝色数据点，优先级最高
        ax.scatter(x, y, color='blue', zorder=10, s=30)

        # 2. 标注每个点的(k, f)
        for xi, yi, k, f in zip(x, y, k_list, f_list):
            ax.text(xi * 1.008, yi * 1.008, f'({int(k)}, {int(f)})', fontsize=12, zorder=11)

        # 3. 绘制红色拟合直线，适配数据范围
        x_fit = np.linspace(np.min(x)*0.92, np.max(x)*1.05, 200)
        y_fit = a * x_fit
        ax.plot(x_fit, y_fit, color='red', linewidth=2.5, label=r'$\omega^2 = \frac{\sigma}{\rho} \cdot k^3$', zorder=5)

        x_min = np.min(x) * 0.9
        x_max = np.max(x) * 1.1
        y_min = np.min(y) * 0.9
        y_max = np.max(y) * 1.1
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # 5. 坐标轴标签
        ax.set_xlabel(r'$k^3$', fontsize=18)
        ax.set_ylabel(r'$\omega^2 \quad (rad/s)^2$', fontsize=18)
        # x轴单位标注
        ax.text(1.01, -0.08, r'$m^{-3}$', transform=ax.transAxes, fontsize=14)

        # 6. 左上角拟合参数文本框
        text_str = f'拟合参数: a = {a_rounded:.8f}\n'
        text_str += f'表面张力系数为:{sigma_fit_rounded:.4f}\n'
        if relative_error_fit is not None:
            text_str += f'相对误差: {relative_error_fit:.2f}%\n'
        text_str += f'相关系数: {r_squared_rounded:.4f}'
        ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='square,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none'))

        # 7. 右下角图例
        ax.legend(fontsize=16, loc='lower right', framealpha=0.9)

        # 8. 网格样式
        plt.grid(True, alpha=0.3, linestyle='-')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        return {
            "a": a_rounded,
            "sigma_fit": sigma_fit_rounded,
            "relative_error_fit": relative_error_fit,
            "r_squared": r_squared_rounded,
            "img_base64": img_base64
        }
    except Exception as e:
        app.logger.error(f"拟合绘图失败: {str(e)}\n{traceback.format_exc()}")
        raise e


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/calculate', methods=['POST'])
def calculate():
    try:
        data = request.get_json()
        required_fields = ['f', 'delta_X_cm', 'H0', 'h', 'L', 'rho']
        for field in required_fields:
            if field not in data or data[field] == '' or data[field] is None:
                raise ValueError(f"必填参数「{field}」不能为空")

        f = float(data['f'])
        delta_X_cm = float(data['delta_X_cm'])
        H0 = float(data['H0'])
        h = float(data['h'])
        L = float(data['L'])
        rho = float(data['rho'])
        sigma0 = float(data['sigma0']) if data.get('sigma0') and data['sigma0'] != '' else None

        result = calculate_surface_tension(f, delta_X_cm, H0, h, L, rho, sigma0)
        return jsonify({"success": True, "data": result})
    except ValueError as e:
        app.logger.warning(f"参数错误: {str(e)}")
        return jsonify({"success": False, "message": f"参数错误：{str(e)}"}), 400
    except Exception as e:
        app.logger.error(f"计算出错: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"success": False, "message": f"计算出错：{str(e)}"}), 500


@app.route('/api/fit', methods=['POST'])
def fit():
    try:
        data = request.get_json()
        experiment_data = data.get('experiment_data', [])
        rho_input = data.get('rho', 1000)
        sigma0_input = data.get('sigma0')

        if not experiment_data or len(experiment_data) < 2:
            raise ValueError("至少需要2组实验数据才能进行拟合")
        try:
            rho = float(rho_input)
            if rho <= 0:
                raise ValueError("液体密度必须大于0")
        except:
            raise ValueError("液体密度必须是有效的数字")

        sigma0 = None
        if sigma0_input and sigma0_input != '':
            try:
                sigma0 = float(sigma0_input)
                if sigma0 <= 0:
                    raise ValueError("标准表面张力系数必须大于0")
            except:
                raise ValueError("标准表面张力系数必须是有效的数字")

        fit_result = fit_and_plot(experiment_data, rho, sigma0)
        return jsonify({"success": True, "data": fit_result})
    except ValueError as e:
        app.logger.warning(f"拟合参数错误: {str(e)}")
        return jsonify({"success": False, "message": f"参数错误：{str(e)}"}), 400
    except Exception as e:
        app.logger.error(f"拟合接口出错: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"success": False, "message": f"拟合绘图出错：{str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)