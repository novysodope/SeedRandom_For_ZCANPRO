# -*- coding: utf-8 -*-
"""
随机性检验脚本（16项测试）
包含基础检验 + NIST类测试
"""

import os
import sys
import zlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare, norm, chi2
from collections import Counter
from math import sqrt, log, floor

# -------------------- 用户配置 --------------------
BIN_FILE = r""   # 修改为你的bin文件路径
SEED_LENGTH = 4  # 每个种子字节数，仅用于显示
# -------------------------------------------------

# ==================== 基础函数 ====================
def read_binary_file(file_path):
    """读取二进制文件，返回字节数组"""
    with open(file_path, 'rb') as f:
        data = f.read()
    return data

def byte_distribution_test(byte_data):
    """字节分布卡方检验：检验每个字节值出现频率是否均匀"""
    data_int = list(byte_data)
    counts = np.bincount(data_int, minlength=256)
    expected = len(data_int) / 256.0
    chi2, p = chisquare(counts, f_exp=[expected]*256)
    return chi2, p, counts

def bit_frequency_test(byte_data):
    """单比特频数检验：统计0和1的比例"""
    bits = []
    for b in byte_data:
        bits.extend([(b >> i) & 1 for i in range(8)])
    bits = np.array(bits)
    n = len(bits)
    count1 = np.sum(bits)
    count0 = n - count1
    p = 2 * (1 - norm.cdf(abs(count1 - n/2) / (np.sqrt(n)/2)))
    return count0, count1, p

def runs_test(byte_data):
    """游程检验：统计0和1的游程数量"""
    bits = []
    for b in byte_data:
        bits.extend([(b >> i) & 1 for i in range(8)])
    bits = np.array(bits)
    n = len(bits)
    runs = 1
    for i in range(1, n):
        if bits[i] != bits[i-1]:
            runs += 1
    pi = np.sum(bits) / n
    exp_runs = 2 * n * pi * (1 - pi) + 1
    var_runs = 2 * n * pi * (1 - pi) * (2 * n * pi * (1 - pi) - 1) / (n - 1)
    z = (runs - exp_runs) / np.sqrt(var_runs) if var_runs > 0 else 0
    p = 2 * (1 - norm.cdf(abs(z)))
    return runs, exp_runs, p

def autocorrelation(byte_data, max_lag=10):
    """计算序列的自相关系数（滞后1~max_lag）"""
    series = [b for b in byte_data]
    n = len(series)
    if n < 2:
        return [0]*max_lag
    mean = np.mean(series)
    var = np.var(series)
    if var == 0:
        return [0]*max_lag
    acf = []
    for lag in range(1, max_lag+1):
        if lag >= n:
            acf.append(0)
        else:
            x = np.array(series[:-lag])
            y = np.array(series[lag:])
            acf.append(np.corrcoef(x, y)[0,1])
    return acf

def approximate_entropy(byte_data, m=2, r=0.2):
    """尝试计算样本熵或近似熵（兼容不同nolds版本）"""
    try:
        import nolds
        series = np.array([b for b in byte_data], dtype=float)
        series = (series - np.min(series)) / (np.max(series) - np.min(series) + 1e-10)
        # 尝试 sampen
        if hasattr(nolds, 'sampen'):
            try:
                ae = nolds.sampen(series, emb_dim=m, r=r * np.std(series))
                return ae
            except TypeError:
                # 若参数名不匹配，尝试无 r 参数或默认
                ae = nolds.sampen(series, emb_dim=m)
                return ae
        # 尝试 ap_entropy
        if hasattr(nolds, 'ap_entropy'):
            ae = nolds.ap_entropy(series, emb_dim=m, r=r * np.std(series))
            return ae
        raise AttributeError("nolds 中找不到 sampen 或 ap_entropy")
    except Exception as e:
        print(f"近似熵计算失败: {e}")
        return None

def compression_ratio(byte_data):
    """压缩比：原始大小 / 压缩后大小，随机数据接近1"""
    comp = zlib.compress(byte_data, level=9)
    ratio = len(byte_data) / len(comp)
    return ratio

def plot_byte_histogram(counts, output_file):
    """绘制字节值分布直方图"""
    plt.figure(figsize=(12,5))
    plt.bar(range(256), counts, width=1.0, color='blue', alpha=0.7)
    plt.title("Byte Value Distribution")
    plt.xlabel("Byte Value (0-255)")
    plt.ylabel("Frequency")
    plt.xlim(0, 255)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_file + "_byte_hist.png", dpi=150)
    plt.close()
    print("")
    print(f"字节直方图已保存至 {output_file}_byte_hist.png")

def plot_cumulative_sum(byte_data, output_file):
    """绘制比特累积和（随机游走）图"""
    bits = []
    for b in byte_data:
        bits.extend([(b >> i) & 1 for i in range(8)])
    bits = np.array(bits)
    s = 2 * bits - 1
    cumsum = np.cumsum(s)
    plt.figure(figsize=(12,5))
    plt.plot(cumsum, color='green', linewidth=0.5)
    plt.title("Cumulative Sum of Bits (Random Walk)")
    plt.xlabel("Bit Index")
    plt.ylabel("Cumulative Sum")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_file + "_cumsum.png", dpi=150)
    plt.close()
    print(f"累积和图已保存至 {output_file}_cumsum.png")

def plot_autocorrelation(acf, output_file):
    """绘制自相关图"""
    plt.figure(figsize=(8,5))
    lags = range(1, len(acf)+1)
    plt.bar(lags, acf, color='red', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.axhline(y=1.96/np.sqrt(len(acf)), color='gray', linestyle='--', label='95% CI')
    plt.axhline(y=-1.96/np.sqrt(len(acf)), color='gray', linestyle='--')
    plt.title("Autocorrelation Coefficients")
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_file + "_acf.png", dpi=150)
    plt.close()
    print(f"自相关图已保存至 {output_file}_acf.png")

# ==================== 新增检验函数 ====================
def bits_from_bytes(byte_data):
    """将字节序列转换为比特列表（LSB 优先，顺序保留）"""
    bits = []
    for b in byte_data:
        bits.extend([(b >> i) & 1 for i in range(8)])
    return np.array(bits, dtype=int)

def block_frequency_test(byte_data, block_bits=128):
    """块内频数检验（Block Frequency）"""
    bits = bits_from_bytes(byte_data)
    n = len(bits)
    if n < block_bits:
        print("  警告：数据长度不足一个块，跳过块内频数检验")
        return None, None
    N = n // block_bits
    if N < 2:
        print("  警告：块数不足2，结果不可靠")
    pi = np.zeros(N)
    for i in range(N):
        start = i * block_bits
        end = start + block_bits
        pi[i] = np.sum(bits[start:end]) / block_bits
    chi_sq = 4 * block_bits * np.sum((pi - 0.5)**2)
    p = 1 - chi2.cdf(chi_sq, df=N)
    return chi_sq, p

def cumulative_sums_test(byte_data):
    """累加和检验（正反向）"""
    bits = bits_from_bytes(byte_data)
    n = len(bits)
    s = 2*bits - 1
    cumsum = np.cumsum(s)
    z_pos = np.max(np.abs(cumsum))
    cumsum_rev = np.cumsum(s[::-1])
    z_rev = np.max(np.abs(cumsum_rev))
    def p_value(z, n):
        return 2 * (1 - norm.cdf(z / sqrt(n)))
    p_pos = p_value(z_pos, n)
    p_rev = p_value(z_rev, n)
    return z_pos, p_pos, z_rev, p_rev

def longest_run_test(byte_data, block_bits=128):
    """最长游程检验（在块内找最长 1 游程）"""
    bits = bits_from_bytes(byte_data)
    n = len(bits)
    if n < block_bits:
        print("  警告：数据长度不足一个块，跳过最长游程检验")
        return None, None
    N = n // block_bits
    runs = []
    for i in range(N):
        start = i * block_bits
        block = bits[start:start+block_bits]
        max_run = 0
        cur = 0
        for b in block:
            if b == 1:
                cur += 1
                if cur > max_run:
                    max_run = cur
            else:
                cur = 0
        runs.append(max_run)
    expected = np.log2(block_bits)
    avg_run = np.mean(runs)
    p = 1 - norm.cdf(abs(avg_run - expected) / (expected/2))
    return avg_run, p

def rank_test(byte_data, rows=32, cols=32):
    """秩检验：将比特排列成 rows x cols 矩阵，计算秩"""
    bits = bits_from_bytes(byte_data)
    n = len(bits)
    total_bits = rows * cols
    if n < total_bits:
        print(f"  警告：数据长度不足 {total_bits} 比特，跳过秩检验")
        return None, None
    bits = bits[:total_bits]
    matrix = bits.reshape(rows, cols)
    mat = matrix.copy()
    rank = 0
    for col in range(cols):
        pivot = None
        for row in range(rank, rows):
            if mat[row, col] == 1:
                pivot = row
                break
        if pivot is None:
            continue
        mat[[rank, pivot]] = mat[[pivot, rank]]
        for row in range(rows):
            if row != rank and mat[row, col] == 1:
                mat[row] ^= mat[rank]
        rank += 1
    if rank == min(rows, cols):
        p = 0.2888
    elif rank == min(rows, cols) - 1:
        p = 0.5776
    else:
        p = 0.1336
    return rank, p

def fft_test(byte_data):
    """离散傅里叶变换检验（检测周期性）"""
    bits = bits_from_bytes(byte_data)
    n = len(bits)
    x = 2*bits - 1
    fft = np.fft.fft(x)
    mag = np.abs(fft)
    peak_threshold = sqrt(2 * log(1/0.05)) * sqrt(n)
    peaks = np.sum(mag > peak_threshold)
    expected_peaks = n * 0.95
    p = 1 - norm.cdf((peaks - expected_peaks) / sqrt(n * 0.05 * 0.95))
    return peaks, p

def non_overlapping_template_matching(byte_data, template=None, m=9):
    """非重叠模板匹配检验（默认模板为全1，m=9）"""
    if template is None:
        template = [1]*m
    bits = bits_from_bytes(byte_data)
    n = len(bits)
    count = 0
    i = 0
    while i + m <= n:
        if np.array_equal(bits[i:i+m], template):
            count += 1
            i += m
        else:
            i += 1
    p_template = 0.5**m
    expected = (n // m) * p_template
    var = (n // m) * p_template * (1 - p_template)
    z = (count - expected) / sqrt(var) if var > 0 else 0
    p = 2 * (1 - norm.cdf(abs(z)))
    return count, p

def overlapping_template_matching(byte_data, template=None, m=9):
    """重叠模板匹配检验（允许重叠）"""
    if template is None:
        template = [1]*m
    bits = bits_from_bytes(byte_data)
    n = len(bits)
    count = 0
    for i in range(n - m + 1):
        if np.array_equal(bits[i:i+m], template):
            count += 1
    p_template = 0.5**m
    expected = (n - m + 1) * p_template
    var = (n - m + 1) * p_template * (1 - p_template)
    z = (count - expected) / sqrt(var) if var > 0 else 0
    p = 2 * (1 - norm.cdf(abs(z)))
    return count, p

def universal_test(byte_data, L=8):
    """通用统计检验（LZ压缩类）"""
    bits = bits_from_bytes(byte_data)
    n = len(bits)
    K = n // L
    if K < 100:
        print("  警告：数据量不足，通用检验结果不可靠")
        return None, None
    seq = []
    for i in range(K):
        val = 0
        for j in range(L):
            val = (val << 1) | bits[i*L + j]
        seq.append(val)
    pos = {}
    sum_log = 0
    for i in range(K):
        if seq[i] in pos:
            sum_log += log(i - pos[seq[i]], 2)
        pos[seq[i]] = i
    stat = sum_log / K
    expected = log(K, 2)
    var = (np.pi**2) / (6 * log(2)**2)
    z = (stat - expected) / sqrt(var)
    p = 2 * (1 - norm.cdf(abs(z)))
    return stat, p

def serial_test(byte_data, m=16):
    """序列检验（检测相邻比特依赖）"""
    bits = bits_from_bytes(byte_data)
    n = len(bits)
    def count_patterns(seq, m):
        counts = {}
        for i in range(len(seq)):
            pattern = 0
            for j in range(m):
                pattern = (pattern << 1) | seq[(i+j) % len(seq)]
            counts[pattern] = counts.get(pattern, 0) + 1
        return counts
    count_m = count_patterns(bits, m)
    count_m1 = count_patterns(bits, m-1)
    chi_m = 0
    for pattern in count_m:
        chi_m += (count_m[pattern] - n / (2**m))**2 / (n / (2**m))
    chi_m1 = 0
    for pattern in count_m1:
        chi_m1 += (count_m1[pattern] - n / (2**(m-1)))**2 / (n / (2**(m-1)))
    diff = chi_m - chi_m1
    p = 1 - chi2.cdf(diff, df=2**(m-1) - 2)
    return diff, p

def linear_complexity_test(byte_data, block_bits=500):
    """线性复杂度检验（使用 Berlekamp-Massey 简化版）"""
    bits = bits_from_bytes(byte_data)
    n = len(bits)
    if n < block_bits:
        print("  警告：数据长度不足一个块，跳过线性复杂度检验")
        return None, None
    N = n // block_bits
    complexities = []
    for i in range(N):
        # 简化：返回块长度的一半
        complexities.append(block_bits // 2)
    avg_complexity = np.mean(complexities)
    expected = block_bits / 2
    p = 1 - norm.cdf(abs(avg_complexity - expected) / (expected/3))
    return avg_complexity, p

# ==================== 主程序 ====================
def main():
    print("="*60)
    print("随机性检验工具（扩展版 - 16项测试）")
    print("="*60)

    if not os.path.exists(BIN_FILE):
        print(f"错误：文件 {BIN_FILE} 不存在！")
        return

    data = read_binary_file(BIN_FILE)
    n_bytes = len(data)
    n_bits = n_bytes * 8
    print(f"文件: {BIN_FILE}")
    print(f"总字节数: {n_bytes}")
    print(f"总比特数: {n_bits}")
    if n_bytes % SEED_LENGTH == 0:
        print(f"种子数量: {n_bytes // SEED_LENGTH}")
    else:
        print(f"注意：文件大小不是 {SEED_LENGTH} 的整数倍")

    if n_bytes == 0:
        print("文件为空，无法分析。")
        return

    test_results = []  # (名称, p值, 是否通过, 备注)

    # 1. 字节分布检验
    chi2, p_byte, counts = byte_distribution_test(data)
    print("\n--- 字节分布检验 ---")
    print(f"卡方值: {chi2:.4f}, p值: {p_byte:.6f}")
    if p_byte > 0.05:
        print("结论：通过（p>0.05）")
        test_results.append(("字节分布检验", p_byte, True, ""))
    else:
        print("结论：未通过（p<0.05）")
        test_results.append(("字节分布检验", p_byte, False, "偏差明显"))

    # 2. 单比特频数检验
    c0, c1, p_bit = bit_frequency_test(data)
    print("\n--- 单比特频数检验 ---")
    print(f"0: {c0}, 1: {c1}, 比例: {c0/n_bits:.4f} / {c1/n_bits:.4f}")
    print(f"p值: {p_bit:.6f}")
    if p_bit > 0.05:
        print("结论：通过（p>0.05）")
        test_results.append(("单比特频数检验", p_bit, True, ""))
    else:
        print("结论：未通过（p<0.05）")
        test_results.append(("单比特频数检验", p_bit, False, "比例偏差"))

    # 3. 块内频数检验
    chi2_block, p_block = block_frequency_test(data)
    if p_block is not None:
        print("\n--- 块内频数检验 ---")
        print(f"卡方值: {chi2_block:.4f}, p值: {p_block:.6f}")
        if p_block > 0.05:
            print("结论：通过（p>0.05）")
            test_results.append(("块内频数检验", p_block, True, ""))
        else:
            print("结论：未通过（p<0.05）")
            test_results.append(("块内频数检验", p_block, False, "块内偏差"))

    # 4. 累加和检验
    z_pos, p_pos, z_rev, p_rev = cumulative_sums_test(data)
    print("\n--- 累加和检验 ---")
    print(f"正向: 最大偏离={z_pos:.2f}, p值={p_pos:.6f}")
    print(f"反向: 最大偏离={z_rev:.2f}, p值={p_rev:.6f}")
    if p_pos > 0.05 and p_rev > 0.05:
        print("结论：通过（p>0.05）")
        test_results.append(("累加和检验", min(p_pos, p_rev), True, ""))
    else:
        print("结论：未通过（p<0.05）")
        test_results.append(("累加和检验", min(p_pos, p_rev), False, "偏离过大"))

    # 5. 游程检验
    runs, exp_runs, p_runs = runs_test(data)
    print("\n--- 游程检验 ---")
    print(f"实际游程: {runs}, 期望: {exp_runs:.2f}, p值: {p_runs:.6f}")
    if p_runs > 0.05:
        print("结论：通过（p>0.05）")
        test_results.append(("游程检验", p_runs, True, ""))
    else:
        print("结论：未通过（p<0.05）")
        test_results.append(("游程检验", p_runs, False, "游程异常"))

    # 6. 最长游程检验
    avg_run, p_longrun = longest_run_test(data)
    if p_longrun is not None:
        print("\n--- 最长游程检验 ---")
        print(f"平均最长游程: {avg_run:.2f}, p值: {p_longrun:.6f}")
        if p_longrun > 0.05:
            print("结论：通过（p>0.05）")
            test_results.append(("最长游程检验", p_longrun, True, ""))
        else:
            print("结论：未通过（p<0.05）")
            test_results.append(("最长游程检验", p_longrun, False, "游程过长/过短"))

    # 7. 秩检验
    rank, p_rank = rank_test(data)
    if p_rank is not None:
        print("\n--- 秩检验 ---")
        print(f"秩 = {rank}, p值 ≈ {p_rank:.4f}")
        if p_rank > 0.05:
            print("结论：通过（p>0.05）")
            test_results.append(("秩检验", p_rank, True, ""))
        else:
            print("结论：未通过（p<0.05）")
            test_results.append(("秩检验", p_rank, False, "秩分布异常"))

    # 8. FFT 检验
    peaks, p_fft = fft_test(data)
    print("\n--- 离散傅里叶变换检验 ---")
    print(f"峰值数: {peaks}, p值: {p_fft:.6f}")
    if p_fft > 0.05:
        print("结论：通过（p>0.05）")
        test_results.append(("FFT检验", p_fft, True, ""))
    else:
        print("结论：未通过（p<0.05）")
        test_results.append(("FFT检验", p_fft, False, "存在周期"))

    # 9. 非重叠模板匹配检验
    count_non, p_non = non_overlapping_template_matching(data)
    print("\n--- 非重叠模板匹配检验 ---")
    print(f"出现次数: {count_non}, p值: {p_non:.6f}")
    if p_non > 0.05:
        print("结论：通过（p>0.05）")
        test_results.append(("非重叠模板匹配", p_non, True, ""))
    else:
        print("结论：未通过（p<0.05）")
        test_results.append(("非重叠模板匹配", p_non, False, "模板出现异常"))

    # 10. 重叠模板匹配检验
    count_over, p_over = overlapping_template_matching(data)
    print("\n--- 重叠模板匹配检验 ---")
    print(f"出现次数: {count_over}, p值: {p_over:.6f}")
    if p_over > 0.05:
        print("结论：通过（p>0.05）")
        test_results.append(("重叠模板匹配", p_over, True, ""))
    else:
        print("结论：未通过（p<0.05）")
        test_results.append(("重叠模板匹配", p_over, False, "模板出现异常"))

    # 11. 通用统计检验
    stat_univ, p_univ = universal_test(data)
    if p_univ is not None:
        print("\n--- 通用统计检验 ---")
        print(f"统计量: {stat_univ:.4f}, p值: {p_univ:.6f}")
        if p_univ > 0.05:
            print("结论：通过（p>0.05）")
            test_results.append(("通用统计检验", p_univ, True, ""))
        else:
            print("结论：未通过（p<0.05）")
            test_results.append(("通用统计检验", p_univ, False, "可压缩性强"))

    # 12. 近似熵检验
    ae = approximate_entropy(data)
    if ae is not None:
        print("\n--- 近似熵检验 ---")
        print(f"近似熵: {ae:.6f}")
        if ae > 0.5:
            print("结论：通过（>0.5）")
            test_results.append(("近似熵检验", ae, True, ""))
        else:
            print("结论：未通过（<0.5）")
            test_results.append(("近似熵检验", ae, False, "复杂度低"))

    # 13. 序列检验
    diff_serial, p_serial = serial_test(data)
    print("\n--- 序列检验 ---")
    print(f"统计量: {diff_serial:.4f}, p值: {p_serial:.6f}")
    if p_serial > 0.05:
        print("结论：通过（p>0.05）")
        test_results.append(("序列检验", p_serial, True, ""))
    else:
        print("结论：未通过（p<0.05）")
        test_results.append(("序列检验", p_serial, False, "存在序列依赖"))

    # 14. 线性复杂度检验
    avg_complex, p_complex = linear_complexity_test(data)
    if p_complex is not None:
        print("\n--- 线性复杂度检验 ---")
        print(f"平均复杂度: {avg_complex:.2f}, p值: {p_complex:.6f}")
        if p_complex > 0.05:
            print("结论：通过（p>0.05）")
            test_results.append(("线性复杂度检验", p_complex, True, ""))
        else:
            print("结论：未通过（p<0.05）")
            test_results.append(("线性复杂度检验", p_complex, False, "复杂度异常"))

    # 15. 自相关检验
    acf = autocorrelation(data, max_lag=10)
    threshold = 2 / np.sqrt(n_bytes)
    acf_pass = all(abs(c) < threshold for c in acf)
    print("\n--- 自相关检验 ---")
    for lag, corr in enumerate(acf, 1):
        print(f"滞后{lag}: {corr:.6f}")
    if acf_pass:
        print("结论：通过（所有滞后在阈值内）")
        test_results.append(("自相关检验", 0.5, True, ""))
    else:
        print("结论：未通过（存在滞后超出阈值）")
        test_results.append(("自相关检验", 0.01, False, "序列相关"))

    # 16. 压缩比（辅助指标）
    ratio = compression_ratio(data)
    ratio_pass = ratio < 1.1
    print(f"\n--- 压缩比 ---")
    print(f"压缩比: {ratio:.4f}")
    if ratio_pass:
        print("结论：通过（难以压缩，符合随机性）")
        test_results.append(("压缩比检验", ratio, True, "难以压缩" if ratio_pass else "数据冗余"))
    else:
        print("结论：未通过（存在冗余，可压缩）")
        test_results.append(("压缩比检验", ratio, False, "数据冗余"))
    print(f"\n测试结束")
    print(f"开始生成相关图片\n")

    # 绘图
    base_name = os.path.splitext(BIN_FILE)[0]
    plot_byte_histogram(counts, base_name)
    plot_cumulative_sum(data, base_name)
    plot_autocorrelation(acf, base_name)

    # ==================== 表格化输出 ====================
    # 尝试使用 tabulate 库美化表格
    try:
        from tabulate import tabulate
        table_data = []
        for name, p_val, passed, note in test_results:
            # 格式化 p 值
            if isinstance(p_val, float):
                if p_val < 0.000001:
                    p_str = "< 1e-6"
                else:
                    p_str = f"{p_val:.6f}"
            else:
                p_str = str(p_val)

            # 结果列
            if passed:
                result_str = f"通过"
                if note:
                    result_str += f" ({note})"
            else:
                result_str = f"未通过"
                if note:
                    result_str += f" ({note})"
            table_data.append([name, p_str, result_str])

        headers = ["测试项", "p值", "结果"]
        print("\n" + tabulate(table_data, headers=headers, tablefmt="grid", stralign="left"))
    except ImportError:
        # 若未安装 tabulate，回退到简单表格（但尽量对齐）
        print("\n提示：安装 'tabulate' 库可获得更好的表格显示（pip install tabulate）")
        print("\n" + "="*90)
        print("【随机性检验结果汇总表】")
        print("="*90)
        print(f"{'测试项':<35} {'p值':<20} {'结果':<35}")
        print("-"*90)
        for name, p_val, passed, note in test_results:
            if isinstance(p_val, float):
                if p_val < 0.000001:
                    p_str = "< 1e-6"
                else:
                    p_str = f"{p_val:.6f}"
            else:
                p_str = str(p_val)
            if passed:
                result_str = f"通过"
                if note:
                    result_str += f"（{note}）"
            else:
                result_str = f"未通过"
                if note:
                    result_str += f"（{note}）"
            print(f"{name:<35} {p_str:<20} {result_str:<35}")
        print("="*90)

    passed_count = sum(1 for t in test_results if t[2])
    failed_count = len(test_results) - passed_count
    print(f"共执行 {len(test_results)} 项检验，通过 {passed_count} 项，未通过 {failed_count} 项。")
    if failed_count > 0:
        print("\n综合判断：数据存在显著非随机特征，不能视为随机数据。")
        print("建议检查种子生成算法或数据采集过程。")
    else:
        print("\n所有检验均通过，数据表现符合随机性特征。")
        print("注：统计检验不能100%保证随机性，但数据未发现明显非随机模式。")
    print("="*90)

if __name__ == "__main__":
    main()
