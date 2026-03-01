import numpy as np
from PIL import Image
import os

# =========================================================
# ĐỌC / LƯU ẢNH
# =========================================================
def doc_anh_duong_dan(duong_dan):
    # Mở ảnh và chuyển sang grayscale (L = luminance)
    # Mục đích: xử lý đơn kênh để dễ làm Fourier
    img = Image.open(duong_dan).convert('L')

    # Chuẩn hoá pixel về [0,1] để tính toán số học ổn định
    return np.array(img) / 255.0


def luu_anh(anh, duong_dan):
    # Chuyển ngược [0,1] → [0,255]
    # Clip để tránh giá trị vượt phạm vi
    anh_luu = np.clip(anh * 255, 0, 255).astype(np.uint8)
    Image.fromarray(anh_luu).save(duong_dan)


# =========================================================
# BIẾN ĐỔI FOURIER 2D THUẦN (THEO CÔNG THỨC)
# =========================================================
_DFT_MAT_CACHE = {}
# Cache để lưu ma trận DFT đã tính → tránh tính lại nhiều lần


def _ma_tran_dft_1d(n, nghich_dao=False):
    """
    Tạo ma trận DFT 1 chiều theo công thức toán học:

    Forward:
        F(u) = Σ f(x) * exp(-j2πux/N)

    Inverse:
        f(x) = (1/N) Σ F(u) * exp(+j2πux/N)

    Không dùng FFT → tính đúng định nghĩa DFT.
    """
    key = (n, bool(nghich_dao))
    if key in _DFT_MAT_CACHE:
        return _DFT_MAT_CACHE[key]

    # vector chỉ số
    u = np.arange(n).reshape(n, 1)
    x = np.arange(n).reshape(1, n)

    # dấu của số mũ
    dau = 1.0 if nghich_dao else -1.0

    # Ma trận Fourier
    W = np.exp(dau * 2j * np.pi * (u * x) / n)

    _DFT_MAT_CACHE[key] = W
    return W


def bien_doi_fourier_2d(anh):
    """
    DFT 2D dùng tính phân ly:

        F = Wm @ img @ Wn

    Nghĩa là:
    - biến đổi Fourier theo hàng
    - rồi theo cột

    Đây chính là định nghĩa DFT 2D:
        F(u,v) = ΣΣ f(x,y) e^{-j2π(ux/M + vy/N)}
    """
    anh = np.asarray(anh, dtype=np.complex128)
    M, N = anh.shape

    Wm = _ma_tran_dft_1d(M, nghich_dao=False)
    Wn = _ma_tran_dft_1d(N, nghich_dao=False)

    return (Wm @ anh) @ Wn


def nghich_dao_bien_doi_fourier_2d(F):
    """
    IDFT 2D:

        img = (W* @ F @ W*) / (M*N)

    W* là liên hợp phức của W
    chia (M*N) để chuẩn hoá.
    """
    F = np.asarray(F, dtype=np.complex128)
    M, N = F.shape

    Wm_i = _ma_tran_dft_1d(M, nghich_dao=True)
    Wn_i = _ma_tran_dft_1d(N, nghich_dao=True)

    anh = ((Wm_i @ F) @ Wn_i) / (M * N)

    # Lấy phần thực (phần ảo chỉ do sai số số học)
    return np.real(anh)


# =========================================================
# MÔ HÌNH HÓA ẢNH MỜ (MOTION BLUR)
# =========================================================
def tao_kernel_chuyen_dong(do_dai=9):
    """
    Tạo kernel motion blur ngang.

    Ý nghĩa vật lý:
    camera di chuyển → ánh sáng bị trải theo hướng chuyển động.

    Kernel là 1 đường thẳng → tích chập sẽ làm nhòe theo hướng đó.
    """
    k = np.zeros((do_dai, do_dai))
    k[do_dai // 2, :] = 1   # vệt ngang
    k /= np.sum(k)          # chuẩn hoá tổng = 1 (bảo toàn năng lượng)
    return k


def them_zero_padding_cho_kernel(kernel, kich_thuoc_anh):
    """
    Khi làm convolution bằng Fourier:

        g = f * h   <=>   G = F . H

    Kernel phải:
    - cùng kích thước ảnh
    - tâm kernel ở (0,0)

    Nhưng kernel thường tạo ở giữa → phải dịch về góc.
    """
    H = np.zeros(kich_thuoc_anh)
    h, w = kernel.shape
    H[:h, :w] = kernel

    # dịch tâm kernel về (0,0)
    H = np.roll(H, -h // 2, axis=0)
    H = np.roll(H, -w // 2, axis=1)

    return H


def lam_mo_bang_dft(anh_goc, kernel):
    """
    Làm mờ bằng convolution trong miền tần số:

        g = f * h

    Fourier:
        G = F . H
    """
    H = them_zero_padding_cho_kernel(kernel, anh_goc.shape)

    F = bien_doi_fourier_2d(anh_goc)
    Hf = bien_doi_fourier_2d(H)

    V = F * Hf   # convolution trong frequency

    return nghich_dao_bien_doi_fourier_2d(V)


# =========================================================
# THÊM NHIỄU GAUSSIAN
# =========================================================
def them_nhieu_gaussian(anh, do_lech_chuan=0.02):
    """
    Nhiễu Gaussian:

        n ~ N(0, sigma^2)

    Mô phỏng nhiễu cảm biến camera.
    """
    nhieu = np.random.normal(0, do_lech_chuan, anh.shape)
    return np.clip(anh + nhieu, 0, 1)


# =========================================================
# LỌC NGƯỢC (INVERSE FILTER)
# =========================================================
def loc_nguoc(anh_mo_nhieu, kernel, epsilon=1e-3):
    """
    Khôi phục lý tưởng:

        F = G / H

    Nhưng nếu |H| ≈ 0 → chia sẽ nổ → khuếch đại nhiễu.

    Giải pháp:
        bỏ qua tần số bị triệt mạnh.
    """
    H = them_zero_padding_cho_kernel(kernel, anh_mo_nhieu.shape)

    V = bien_doi_fourier_2d(anh_mo_nhieu)
    Hf = bien_doi_fourier_2d(H)

    nguong = epsilon * np.max(np.abs(Hf))

    U = np.zeros_like(V)

    mask = np.abs(Hf) >= nguong
    U[mask] = V[mask] / Hf[mask]

    return np.clip(nghich_dao_bien_doi_fourier_2d(U), 0, 1)


# =========================================================
# LỌC WIENER
# =========================================================
def loc_wiener(anh_mo_nhieu, kernel, he_so_K=0.01):
    """
    Wiener filter tối ưu theo MSE.

    Công thức:

        F = (H* / (|H|^2 + K)) G

    K = tỉ lệ nhiễu / tín hiệu

    Ý nghĩa:
    - khi |H| lớn → gần inverse filter
    - khi |H| nhỏ → giảm khuếch đại nhiễu
    """
    H = them_zero_padding_cho_kernel(kernel, anh_mo_nhieu.shape)

    V = bien_doi_fourier_2d(anh_mo_nhieu)
    Hf = bien_doi_fourier_2d(H)

    H_conj = np.conj(Hf)

    U = (H_conj / (np.abs(Hf)**2 + he_so_K)) * V

    return np.clip(nghich_dao_bien_doi_fourier_2d(U), 0, 1)


# =========================================================
# ĐÁNH GIÁ CHẤT LƯỢNG
# =========================================================
def sai_so_binh_phuong_trung_binh(anh1, anh2):
    """
    MSE = trung bình bình phương sai khác pixel.
    """
    return np.mean((anh1 - anh2)**2)


def psnr(anh_goc, anh_khoi_phuc):
    """
    PSNR đo chất lượng khôi phục.

        PSNR = 10 log10(MAX^2 / MSE)

    Ảnh chuẩn hoá MAX = 1
    """
    mse = sai_so_binh_phuong_trung_binh(anh_goc, anh_khoi_phuc)
    if mse == 0:
        return 99
    return 10 * np.log10(1 / mse)