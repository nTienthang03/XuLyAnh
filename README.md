# XuLyAnh
# Giáo viên hướng dẫn: ThS. Đặng Thị Hiên 
1. Tên đề tài : Khôi phục ảnh bị nhiễu và mờ
2. Nội dung các phần thuyết minh và tính toán
- Mô hình hóa ảnh mờ và nhiễu.
- Cài đặt lọc ngược và lọc Wiener.
- So sánh hiệu quả các phương pháp.
- Mã nguồn chương trình.
- Bộ ảnh đầu vào và ảnh kết quả.
- File báo cáo bài tập lớn.
3. Các sản phẩm, kết quả :
-	Thuyết minh báo cáo (1 quyển)
-	Code chạy chương trình 

# Tổng quan

Workspace này là bài thực hành **Xử lý ảnh**: mô phỏng ảnh bị **mờ chuyển động (motion blur)** và **nhiễu Gaussian**, sau đó khôi phục bằng **lọc ngược (Inverse filter)** và **lọc Wiener** trong miền tần số.

Điểm chính:

- Biến đổi Fourier 2D được cài đặt **thuần theo định nghĩa DFT** (dùng ma trận $W$), không dùng FFT; có cache ma trận DFT để tránh tính lại.
- Kernel motion blur là vệt ngang (có thể chỉnh độ dài).
- Hỗ trợ đánh giá chất lượng bằng **MSE** và **PSNR**.

## Cấu trúc

- `code.py`: các hàm xử lý ảnh (DFT/IDFT, blur, noise, inverse/Wiener, PSNR).
- `ketqua/`: ảnh kết quả mẫu (gốc, mờ, mờ+nhiễu, khôi phục).
- `baocaoo.docx`: báo cáo.

Các ảnh mẫu trong `ketqua/`:

- `1_goc.png`: ảnh gốc
- `2_mo.png`: ảnh sau làm mờ
- `3_mo_nhieu.png`: ảnh mờ + nhiễu
- `4_nguoc.png`: khôi phục bằng lọc ngược
- `5_wiener.png`: khôi phục bằng lọc Wiener

## Yêu cầu

- Python 3
- Thư viện: `numpy`, `Pillow`

Cài đặt nhanh:

```bash
py -m pip install numpy pillow
```

## Cách chạy (demo)

`code.py` hiện là module hàm (chưa có `main`). Bạn có thể chạy nhanh bằng cách mở Python và chạy đoạn sau (ví dụ dùng `ketqua/1_goc.png` làm input):

```python
import os
import numpy as np

from code import (
	doc_anh_duong_dan, luu_anh,
	tao_kernel_chuyen_dong, lam_mo_bang_dft,
	them_nhieu_gaussian, loc_nguoc, loc_wiener,
	psnr
)

# (Tuỳ chọn) cố định nhiễu để dễ so sánh
np.random.seed(0)

inp = os.path.join('ketqua', '1_goc.png')
out_dir = 'ketqua'
os.makedirs(out_dir, exist_ok=True)

anh_goc = doc_anh_duong_dan(inp)
kernel = tao_kernel_chuyen_dong(do_dai=9)

anh_mo = lam_mo_bang_dft(anh_goc, kernel)
anh_mo_nhieu = them_nhieu_gaussian(anh_mo, do_lech_chuan=0.02)

anh_nguoc = loc_nguoc(anh_mo_nhieu, kernel, epsilon=1e-3)
anh_wiener = loc_wiener(anh_mo_nhieu, kernel, he_so_K=0.01)

luu_anh(anh_goc, os.path.join(out_dir, '1_goc.png'))
luu_anh(anh_mo, os.path.join(out_dir, '2_mo.png'))
luu_anh(anh_mo_nhieu, os.path.join(out_dir, '3_mo_nhieu.png'))
luu_anh(anh_nguoc, os.path.join(out_dir, '4_nguoc.png'))
luu_anh(anh_wiener, os.path.join(out_dir, '5_wiener.png'))

print('PSNR (Inverse):', psnr(anh_goc, anh_nguoc))
print('PSNR (Wiener) :', psnr(anh_goc, anh_wiener))
```

## Ghi chú hiệu năng

Vì DFT 2D được tính bằng ma trận theo công thức, thời gian chạy sẽ tăng rất nhanh theo kích thước ảnh. Nếu chạy chậm, hãy thử ảnh nhỏ (ví dụ 128×128 hoặc 256×256).
