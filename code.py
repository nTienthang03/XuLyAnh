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
    return loc_nguoc_tu_tan_so(V, Hf, epsilon=epsilon)


def loc_nguoc_tu_tan_so(V, Hf, epsilon=1e-3):
    """Loc nguoc khi da co san V=DFT(anh) va Hf=DFT(kernel pad).

    Giup tai su dung V/Hf cho nhieu bo loc (inverse + wiener) ma khong can tinh DFT lap.
    """
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
    return loc_wiener_tu_tan_so(V, Hf, he_so_K=he_so_K)


def loc_wiener_tu_tan_so(V, Hf, he_so_K=0.03):
    """Wiener filter khi da co san V=DFT(anh) va Hf=DFT(kernel pad."""
    H_conj = np.conj(Hf)
    U = (H_conj / (np.abs(Hf) ** 2 + he_so_K)) * V
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


def xu_ly_anh_motion_blur(
    duong_dan_anh_goc="anh_goc.jpg",
    thu_muc_ket_qua="ketqua",
    do_dai_kernel=9,
    do_lech_chuan_nhieu=0.05,
    epsilon_loc_nguoc=None,
    he_so_K_wiener=0.06,
    random_seed=0,
):
    if random_seed is not None:
        np.random.seed(int(random_seed))

    if not os.path.exists(duong_dan_anh_goc):
        raise FileNotFoundError(
            f"Khong tim thay anh dau vao: {duong_dan_anh_goc}. "
            "Hay dat file anh trong cung thu muc voi code.py hoac truyen duong dan."
        )

    os.makedirs(thu_muc_ket_qua, exist_ok=True)

    anh_goc = doc_anh_duong_dan(duong_dan_anh_goc)
    M, N = anh_goc.shape
    if max(M, N) > 256:
        print(
            "Canh bao: anh kich thuoc lon (",
            f"{M}x{N}",
            ") - DFT thuan se rat cham."
        )

    kernel = tao_kernel_chuyen_dong(do_dai=do_dai_kernel)

    anh_mo = np.clip(lam_mo_bang_dft(anh_goc, kernel), 0, 1)
    anh_mo_nhieu = them_nhieu_gaussian(anh_mo, do_lech_chuan=do_lech_chuan_nhieu)

    # Chon tham so "chuan" theo ly thuyet:
    # - Inverse filter: epsilon la nguong cat cac tan so co |H| qua nho (tranh no nhieu).
    #   Với kernel đã chuẩn hoá, max(|H|) xấp xỉ 1.
    #   Thuc te epsilon thuong can lon hon sigma (vi inverse rat nhay voi nhieu).
    #   He so nhan (c) la kinh nghiem: c cang lon thi cang on dinh (it no nhieu) nhung mat nhieu tan so cao.
    #   Trong code nay dang dung epsilon ~ 30*sigma (co clip) de on dinh hon khi co nhieu.
    # - Wiener: K ~ (cong suat nhieu) / (cong suat tin hieu)
    #   xấp xỉ sigma^2 / Var(anh)
    if epsilon_loc_nguoc is None:
        epsilon_loc_nguoc = float(np.clip(50* do_lech_chuan_nhieu, 1e-3, 2e-1))

    if he_so_K_wiener is None:
        noise_var = float(do_lech_chuan_nhieu) ** 2
        signal_var = float(np.var(anh_mo))
        he_so_K_wiener = noise_var / max(signal_var, 1e-12)
        he_so_K_wiener = float(np.clip(he_so_K_wiener, 1e-6, 1e-1))

    # Tinh trong mien tan so 1 lan de dung cho ca 2 bo loc
    H = them_zero_padding_cho_kernel(kernel, anh_mo_nhieu.shape)
    V = bien_doi_fourier_2d(anh_mo_nhieu)
    Hf = bien_doi_fourier_2d(H)

    anh_nguoc = loc_nguoc_tu_tan_so(V, Hf, epsilon=epsilon_loc_nguoc)
    anh_wiener = loc_wiener_tu_tan_so(V, Hf, he_so_K=he_so_K_wiener)

    duong_dan_1 = os.path.join(thu_muc_ket_qua, "1_goc.png")
    duong_dan_2 = os.path.join(thu_muc_ket_qua, "2_mo.png")
    duong_dan_3 = os.path.join(thu_muc_ket_qua, "3_mo_nhieu.png")
    duong_dan_4 = os.path.join(thu_muc_ket_qua, "4_nguoc.png")
    duong_dan_5 = os.path.join(thu_muc_ket_qua, "5_wiener.png")

    luu_anh(anh_goc, duong_dan_1)
    luu_anh(anh_mo, duong_dan_2)
    luu_anh(anh_mo_nhieu, duong_dan_3)
    luu_anh(anh_nguoc, duong_dan_4)
    luu_anh(anh_wiener, duong_dan_5)

    mse_nguoc = sai_so_binh_phuong_trung_binh(anh_goc, anh_nguoc)
    mse_wiener = sai_so_binh_phuong_trung_binh(anh_goc, anh_wiener)
    psnr_nguoc = psnr(anh_goc, anh_nguoc)
    psnr_wiener = psnr(anh_goc, anh_wiener)

    print("Da luu ket qua vao:", os.path.abspath(thu_muc_ket_qua))
    print(
        "Thong so:",
        f"kernel={do_dai_kernel}",
        f"sigma={do_lech_chuan_nhieu}",
        f"epsilon={epsilon_loc_nguoc}",
        f"K={he_so_K_wiener}",
    )

    # In ra bang so sanh (de copy vao bao cao)
    print("\n{:<20} {:>12} {:>12}".format("Phuong phap", "MSE", "PSNR(dB)"))
    print("{:<20} {:>12.6f} {:>12.2f}".format("Loc nguoc", mse_nguoc, psnr_nguoc))
    print("{:<20} {:>12.6f} {:>12.2f}".format("Loc Wiener", mse_wiener, psnr_wiener))

    # Luu ra file text de nop/doi chieu
    duong_dan_bang = os.path.join(thu_muc_ket_qua, "so_sanh_mse_psnr.txt")
    with open(duong_dan_bang, "w", encoding="utf-8") as f:
        f.write(f"kernel={do_dai_kernel}\n")
        f.write(f"sigma={do_lech_chuan_nhieu}\n")
        f.write(f"epsilon={epsilon_loc_nguoc}\n")
        f.write(f"K={he_so_K_wiener}\n\n")
        f.write("Phuong phap\tMSE\tPSNR(dB)\n")
        f.write(f"Loc nguoc\t{mse_nguoc:.6f}\t{psnr_nguoc:.2f}\n")
        f.write(f"Loc Wiener\t{mse_wiener:.6f}\t{psnr_wiener:.2f}\n")

    return {
        "anh_goc": anh_goc,
        "anh_mo": anh_mo,
        "anh_mo_nhieu": anh_mo_nhieu,
        "anh_nguoc": anh_nguoc,
        "anh_wiener": anh_wiener,
        "mse_nguoc": mse_nguoc,
        "mse_wiener": mse_wiener,
        "psnr_nguoc": psnr_nguoc,
        "psnr_wiener": psnr_wiener,
        "duong_dan_bang": duong_dan_bang,
    }


if __name__ == "__main__":
    # Chay mac dinh de xu ly file anh_goc.jpg (cung thu muc voi code.py)
    # Co the truyen duong dan anh qua tham so dong lenh:
    #   python code.py duong_dan\toi\anh.jpg
    import sys

    duong_dan_anh = sys.argv[1] if len(sys.argv) > 1 else "anh_goc.jpg"
    xu_ly_anh_motion_blur(duong_dan_anh_goc=duong_dan_anh)
