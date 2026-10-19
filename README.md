# CSE457 - XỬ LÝ ÂM THANH VÀ TIẾNG NÓI
## BÀI THỰC HÀNH SỐ 3 (LAB 3): NHẬN DẠNG TIẾNG NÓI NÂNG CAO VỚI HMM
**Trường Đại học Thủy Lợi - Khoa Công nghệ Thông tin - Bộ môn Trí tuệ Nhân tạo**

* **Sinh viên:** Nguyễn Thị Linh
* **MSSV:** 2351260663
* **Lớp:** 65TTNT

---

## 1. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
Lab03_2351260663_NguyenThiLinh/
├── Lab 3.pdf                           # Đề bài thực hành gốc của học phần CSE457
├── Lab03.ipynb                         # Jupyter Notebook hoàn chỉnh (đã chạy toàn bộ kèm output & hình ảnh)
├── Lab03.py                            # Toàn bộ mã nguồn Python thực thi độc lập từ đầu đến cuối
├── report_Lab03.md                     # Báo cáo kỹ thuật chi tiết (kết quả, phân tích, trả lời 10 câu hỏi)
├── README.md                           # Tài liệu hướng dẫn sử dụng và cấu trúc dự án
├── lexicon.txt                         # Từ điển phát âm mini cho 5 lệnh (WORD PHONE_1 ... PHONE_K)
├── corpus.txt                          # Tập ngữ liệu huấn luyện Bigram Language Model
├── results.json                        # Tổng hợp số liệu các thí nghiệm dưới định dạng JSON
├── results.csv                         # Bảng so sánh chỉ số định lượng giữa DTW và HMM
├── generate_dataset.py                 # Mã nguồn tổng hợp dữ liệu âm thanh kiểm thử và huấn luyện
├── build_notebook.py                   # Script tự động xây dựng và thực thi Lab03.ipynb
├── dataset/                            # Bộ dữ liệu âm thanh 16kHz WAV mono
│   ├── train/                          # 30 file huấn luyện (6 file/lớp x 5 lệnh: mo, dong, len, xuong, dung)
│   ├── test/                           # 10 file kiểm thử sạch (2 file/lớp x 5 lệnh)
│   ├── test_spk2/                      # 10 file kiểm thử khác người nói (Speaker Mismatch E5)
│   └── test_noisy/                     # 10 file kiểm thử có nhiễu nền SNR 12dB (Environment Mismatch E5)
├── models/                             # Thư mục lưu trữ 5 mô hình Whole-word HMM đã huấn luyện
│   ├── hmm_mo.pkl
│   ├── hmm_dong.pkl
│   ├── hmm_len.pkl
│   ├── hmm_xuong.pkl
│   └── hmm_dung.pkl
└── figures/                            # Toàn bộ 11 biểu đồ trực quan hóa độ phân giải cao (300 DPI)
    ├── fig1_mfcc_delta_heatmaps.png
    ├── fig2_viterbi_alignment_mo.png
    ├── fig3_viterbi_alignment_dong.png
    ├── fig4_confusion_matrix_hmm.png
    ├── fig5_dtw_vs_hmm_comparison.png
    ├── fig6_exp_n_states.png
    ├── fig7_exp_mfcc_vs_delta.png
    ├── fig8_exp_covariance_type.png
    ├── fig9_exp_mismatch_robustness.png
    ├── fig10_exp_lm_reranking.png
    └── fig11_score_distribution_rejection.png
```

---

## 2. YÊU CẦU MÔI TRƯỜNG VÀ CÀI ĐẶT

### 2.1. Cài đặt các thư viện cần thiết
Hệ thống yêu cầu Python 3.8+ và các thư viện xử lý âm thanh, học máy:
```bash
pip install numpy scipy matplotlib seaborn librosa soundfile scikit-learn hmmlearn gtts
```

---

## 3. HƯỚNG DẪN THỰC THI

### 3.1. Chạy toàn bộ pipeline bằng Python Script
Chạy file [Lab03.py](Lab03.py) để thực hiện toàn bộ các bước từ trích xuất đặc trưng, huấn luyện HMM, đánh giá DTW baseline, chạy các thí nghiệm E1-E6, xuất đồ thị và lưu mô hình:
```bash
python Lab03.py
```

### 3.2. Mở và kiểm tra Jupyter Notebook
Mở file [Lab03.ipynb](Lab03.ipynb) trong Visual Studio Code hoặc Jupyter Notebook / JupyterLab:
```bash
jupyter notebook Lab03.ipynb
```
*Tất cả các cell trong `Lab03.ipynb` đã được chạy sẵn và lưu đầy đủ kết quả số liệu, ma trận nhầm lẫn và biểu đồ đồ họa sắc nét.*

---

## 4. TÓM TẮT KẾT QUẢ THỰC NGHIỆM CHÍNH

1. **Hiệu năng nhận dạng trên tập Test Sạch:**
   - **HMM (Lab 3):** Độ chính xác **100.0%**.
   - **DTW (Lab 2):** Độ chính xác **100.0%**.
2. **Độ trễ suy luận thời gian thực (Inference Latency):**
   - **HMM:** **6.42 ms / mẫu**.
   - **DTW:** **413.82 ms / mẫu**.
   - *HMM nhanh gấp **64.5 lần** so với DTW nhờ cấu trúc Forward/Viterbi ma trận nhỏ gọn.*
3. **Thí nghiệm Bắt buộc:**
   - **E1 (Số trạng thái):** $n\_states = 5$ là cấu hình tối ưu nhất cho cấu trúc ngữ âm từ đơn tiếng Việt.
   - **E2 (Đặc trưng động):** MFCC 39 chiều (+$\Delta$, +$\Delta^2$) giúp tăng khoảng cách Margin giữa Top-1 và Top-2 thêm hơn 35%.
   - **E3 (Covariance type):** Diagonal Covariance (195 tham số) ổn định và chống quá khớp vượt trội so với Full Covariance (3900 tham số).
   - **E5 (Mismatch):** Giữ vững 100% khi đổi người nói; giảm xuống 60% khi gặp nhiễu nền SNR 12dB.
   - **E6 (LM & MAP Re-ranking):** Tích hợp thành công Bigram LM có làm mịn Add-k ($k=0.5$).
   - **Mở rộng (OOV Rejection):** Xác định ngưỡng tối ưu $\theta_{\text{reject}} = -71.18$ để từ chối các từ lạ ngoài từ vựng.
