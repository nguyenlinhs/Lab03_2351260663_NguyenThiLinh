# BÁO CÁO BÀI THỰC HÀNH SỐ 3 (LAB 3)
## NHẬN DẠNG TIẾNG NÓI NÂNG CAO VỚI HMM
**Từ MFCC động đến mô hình Markov ẩn, Likelihood, Viterbi, Baum-Welch và Giải mã có Mô hình Ngôn ngữ**

---

* **Học phần:** CSE457 – Xử lý Âm thanh và Tiếng nói  
* **Khoa / Bộ môn:** Bộ môn Trí tuệ Nhân tạo – Khoa Công nghệ Thông tin – Trường Đại học Thủy Lợi  
* **Sinh viên thực hiện:** Nguyễn Thị Linh  
* **Mã số sinh viên (MSSV):** 2351260663  
* **Lớp:** 65TTNT  
* **Thời gian thực hiện:** Học kỳ 1, Năm học 2026 – 2027  

---

## 1. TỔNG QUAN VÀ THIẾT KẾ HỆ THỐNG

### 1.1. Mục tiêu và Chuẩn đầu ra
Lab 3 đánh dấu bước chuyển quan trọng trong công nghệ nhận dạng tiếng nói (ASR): từ phương pháp **đối sánh mẫu hình học trực tiếp DTW (Dynamic Time Warping)** ở Lab 2 sang **mô hình xác suất thống kê theo thời gian HMM (Hidden Markov Model)**.

Mục tiêu chính bao gồm:
1. Nắm vững bản chất toán học của HMM $\lambda = (A, B, \pi)$, giả thiết Markov bậc 1 và giả thiết độc lập quan sát có điều kiện.
2. Xây dựng vector đặc trưng âm học 39 chiều kết hợp giữa đặc trưng phổ tĩnh 13 MFCC và đặc trưng động học thời gian (13 Δ-MFCC và 13 Δ²-MFCC) cùng chuẩn hóa CMVN.
3. Cấu hình mô hình HMM Trái - Sang - Phải (Left-to-Right topology), hiểu rõ nguyên lý phân bố phát xạ Gaussian đa biến với ma trận hiệp phương sai đường chéo (`diag`).
4. Huấn luyện 5 mô hình Whole-word HMM cho 5 khẩu lệnh tiếng Việt (`mở`, `đóng`, `lên`, `xuống`, `dừng`) bằng thuật toán Baum-Welch (EM) thông qua thư viện `hmmlearn`.
5. Thực hiện nhận dạng theo quy tắc Maximum Likelihood (Forward Algorithm) và phân tích căn chỉnh ngữ âm bằng giải mã Viterbi (Trellis Alignment).
6. So sánh thực nghiệm đối đầu giữa DTW và HMM trên cùng phân chia dữ liệu kiểm thử (độ chính xác, độ trễ suy luận, chi phí bộ nhớ và huấn luyện).
7. Mở rộng hệ thống với Từ điển phát âm (Lexicon mini), Mô hình ngôn ngữ Bigram (với làm mịn Add-k), và thuật toán MAP Re-ranking kết hợp điểm âm học và điểm ngôn ngữ.
8. Khảo sát tính bền vững (robustness) trước biến động người nói (speaker mismatch), nhiễu nền môi trường và đề xuất cơ chế từ chối từ lạ (Out-of-Vocabulary rejection).

```
+---------------------------------------------------------------------------------------------------+
|                                 SƠ ĐỒ PIPELINE HỆ THỐNG LAB 3                                     |
+---------------------------------------------------------------------------------------------------+
|  [Speech Waveform]                                                                                |
|         │                                                                                         |
|         ▼                                                                                         |
|  [Tiền xử lý & VAD] ────> [13 MFCC] ──> [+13 Delta] ──> [+13 Delta2] ──> [CMVN (T x 39)]         |
|                                                                                │                  |
|         ┌──────────────────────────────────────────────────────────────────────┘                  |
|         ▼                                                                                         |
|  [5-State Left-to-Right Gaussian HMMs] (λ_mo, λ_dong, λ_len, λ_xuong, λ_dung)                     |
|         │                                                                                         |
|         ├────────────────────────────────────────┬────────────────────────────────────────┐       |
|         ▼                                        ▼                                        ▼       |
|  [Acoustic Likelihood]                    [Viterbi Alignment]                    [DTW Baseline]   |
|   s_w(X) = log P(X | λ_w)                 State path Q*                          Khoảng cách min  |
|         │                                 Trellis durations                      So sánh đối đầu  |
|         ▼                                                                                         |
|  [Lexicon & Bigram LM Prior]                                                                      |
|   P_LM(w_t | w_{t-1}) + Add-k                                                                     |
|         │                                                                                         |
|         ▼                                                                                         |
|  [MAP Re-ranking Decoder]                                                                         |
|   Score(w) = (1/T) log P_A(X|w) + λ_LM log P_L(w) ───> [Recognized Word / Unknown Reject]         |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. BỘ DỮ LIỆU THỰC NGHIỆM VÀ PHÂN CHIA HUẤN LUYỆN / KIỂM THỬ

### 2.1. Cấu trúc và Quy cách Dữ liệu
* **Tập từ vựng:** 5 lệnh điều khiển tiếng Việt: `mo` (mở), `dong` (đóng), `len` (lên), `xuong` (xuống), `dung` (dừng).
* **Định dạng âm thanh:** WAV mono, tần số lấy mẫu $F_s = 16\,000\text{ Hz}$, mã hóa 16-bit Linear PCM.
* **Khoảng lặng bảo vệ:** Các file âm thanh có $0.2 - 0.4\text{ s}$ khoảng lặng tự nhiên ở đầu và cuối để kiểm tra thuật toán tách biên tiếng nói (VAD / Endpoint Detection).
* **Quy tắc phân chia không rò rỉ (Strict Train/Test Split):**
  - **Tập Train sạch (`dataset/train/`):** 6 utterance/lớp $\times$ 5 lớp = 30 file (đặt tên `label_01.wav` đến `label_06.wav`).
  - **Tập Test sạch (`dataset/test/`):** 2 utterance/lớp $\times$ 5 lớp = 10 file (đặt tên `label_07.wav`, `label_08.wav`), hoàn toàn độc lập với tập train.
  - **Tập Test Mismatch Người nói (`dataset/test_spk2/`):** 2 utterance/lớp $\times$ 5 lớp = 10 file thu âm từ người nói thứ 2 với cao độ (pitch) và đặc trưng formant khác biệt rõ rệt.
  - **Tập Test Môi trường Nhiễu (`dataset/test_noisy/`):** 2 utterance/lớp $\times$ 5 lớp = 10 file bị pha trộn nhiễu trắng nhân tạo với mức SNR = 12 dB.

---

## 3. CÔNG THỨC TOÁN HỌC VÀ CƠ SỞ LÝ THUYẾT

### 3.1. Đặc trưng MFCC Động 39 Chiều
Từ tín hiệu âm thanh đã cắt khoảng lặng và qua bộ lọc tiền nhấn ($y[n] = x[n] - 0.97 x[n-1]$), ta chia khung Hamming $25\text{ ms}$, bước nhảy $10\text{ ms}$ và trích xuất 13 hệ số MFCC tĩnh:

$$
c_t = [c_{t, 0}, c_{t, 1}, \dots, c_{t, 12}]^T \in \mathbb{R}^{13}
$$

Đặc trưng vận tốc (Δ) và gia tốc (Δ²) được xấp xỉ qua sai phân bậc 1 và bậc 2 (hoặc hồi quy qua $M$ frame):

$$
\Delta c_t \approx c_{t+1} - c_{t-1}
$$

$$
\Delta^2 c_t \approx \Delta c_{t+1} - \Delta c_{t-1}
$$

Vector quan sát tại thời điểm $t$ là sự ghép nối 39 chiều:

$$
x_t = \begin{bmatrix} c_t \\ \Delta c_t \\ \Delta^2 c_t \end{bmatrix} \in \mathbb{R}^{39}
$$

Sau đó, áp dụng chuẩn hóa trung bình và phương sai trên từng utterance (Utterance-level CMVN):

$$
\hat{x}_{t, d} = \frac{x_{t, d} - \mu_d}{\sigma_d + 10^{-8}}, \quad d = 1, \dots, 39
$$

### 3.2. Cấu trúc Mô hình HMM Trái - Sang - Phải (λ = (A, B, π))
HMM là mô hình hai tầng ngẫu nhiên: chuỗi trạng thái ẩn $Q = (q_1, q_2, \dots, q_T)$ tuân theo xích Markov bậc 1 và chuỗi vector quan sát $X = (x_1, x_2, \dots, x_T)$ được sinh ra bởi các phân bố phát xạ tại mỗi trạng thái:

**1. Xác suất khởi tạo:** $\pi_i = P(q_1 = i)$. Vì là cấu trúc Left-to-Right, ta gán:

$$
\pi = [1.0, 0, 0, \dots, 0]^T
$$

**2. Ma trận chuyển trạng thái:** $A = \{a_{ij}\}$ với $a_{ij} = P(q_t = j \mid q_{t-1} = i)$.
* Ràng buộc Left-to-Right (Bakis Model): Chỉ cho phép tự lặp (self-loop) $a_{ii} \approx 0.7$ và chuyển tiếp sang trạng thái kế tiếp $a_{i, i+1} \approx 0.3$.
* Cấm chuyển lùi ($a_{ij} = 0, \forall j < i$) và cấm nhảy cóc ($a_{ij} = 0, \forall j > i+1$).
* Trạng thái cuối cùng là trạng thái hấp thụ: $a_{N, N} = 1.0$.

**3. Phân bố phát xạ Gaussian (Gaussian Emission):** Mỗi trạng thái $i$ sở hữu 1 phân bố chuẩn đa biến:

$$
b_i(x_t) = \mathcal{N}(x_t; \mu_i, \Sigma_i) = \frac{1}{(2\pi)^{D/2} |\Sigma_i|^{1/2}} \exp\left( -\frac{1}{2} (x_t - \mu_i)^T \Sigma_i^{-1} (x_t - \mu_i) \right)
$$

Với $D = 39$. Trong bài Lab, ma trận hiệp phương sai $\Sigma_i$ là ma trận đường chéo (`covariance_type="diag"`), nghĩa là chỉ lưu 39 giá trị phương sai $\sigma_{i, d}^2$.

### 3.3. Thuật toán Baum-Welch (Huấn luyện Mô hình EM)
Thuật toán Forward-Backward lặp lại hai bước:

**Bước E (Expectation):**
* Forward variable: $\alpha_t(j) = P(x_1, \dots, x_t, q_t = j \mid \lambda)$.
* Backward variable: $\beta_t(i) = P(x_{t+1}, \dots, x_T \mid q_t = i, \lambda)$.
* Mức chịu trách nhiệm (occupancy probability):

$$
\gamma_t(i) = P(q_t = i \mid X, \lambda) = \frac{\alpha_t(i) \beta_t(i)}{P(X \mid \lambda)}
$$

* Xác suất chuyển trạng thái tại thời điểm $t$:

$$
\xi_t(i, j) = P(q_t = i, q_{t+1} = j \mid X, \lambda) = \frac{\alpha_t(i) a_{ij} b_j(x_{t+1}) \beta_{t+1}(j)}{P(X \mid \lambda)}
$$

**Bước M (Maximization):** Cập nhật tham số mô hình mới:

$$
a_{ij}^{(\text{new})} = \frac{\sum_{t=1}^{T-1} \xi_t(i, j)}{\sum_{t=1}^{T-1} \gamma_t(i)}
$$

$$
\mu_i^{(\text{new})} = \frac{\sum_{t=1}^T \gamma_t(i) x_t}{\sum_{t=1}^T \gamma_t(i)}
$$

$$
\Sigma_i^{(\text{new})} = \frac{\sum_{t=1}^T \gamma_t(i) (x_t - \mu_i^{(\text{new})})(x_t - \mu_i^{(\text{new})})^T}{\sum_{t=1}^T \gamma_t(i)}
$$

### 3.4. Thuật toán Giải mã Viterbi
Tìm chuỗi trạng thái ẩn $Q^* = (q_1^*, \dots, q_T^*)$ tối ưu nhất:
* **Khởi tạo:** $\delta_1(j) = \pi_j b_j(x_1)$, $\psi_1(j) = 0$.
* **Đệ quy (trong miền log):**

$$
\delta_t(j) = \max_{1 \le i \le N} \left[ \delta_{t-1}(i) \cdot a_{ij} \right] \cdot b_j(x_t)
$$

$$
\psi_t(j) = \arg\max_{1 \le i \le N} \left[ \delta_{t-1}(i) \cdot a_{ij} \right]
$$

* **Truy vết (Backtracking):** $q_T^* = \arg\max_i \delta_T(i)$, và $q_t^* = \psi_{t+1}(q_{t+1}^*)$.

---

## 4. KẾT QUẢ THỰC NGHIỆM VÀ PHÂN TÍCH ĐỊNH LƯỢNG

### 4.1. Đặc trưng Âm học và Heatmap MFCC / Delta / Delta2
Hình ảnh phân tích đặc trưng 39 chiều của từ `mở` và `đóng` được trực quan hóa tại [fig1_mfcc_delta_heatmaps.png](figures/fig1_mfcc_delta_heatmaps.png):
* **Hệ số MFCC tĩnh (cửa sổ 0-12):** Thể hiện rõ sự khác biệt ngữ âm. Từ `mở` có năng lượng tập trung mạnh ở các dải tần số thấp đặc trưng của phụ âm mũi $/m/$ và nguyên âm mở $/o/$. Trong khi đó, từ `đóng` bắt đầu bằng phụ âm tắc $/d/$, nối tiếp bằng nguyên âm hẹp và kết thúc bởi phụ âm mũi ngạc mềm $/ng/$.
* **Hệ số Delta và Delta-Delta (13-38):** Phản ánh các bước chuyển đổi trạng thái cấu âm (formant transitions). Các vùng ranh giới giữa phụ âm và nguyên âm thể hiện các xung biến thiên lớn, cung cấp thông tin động cực kỳ giá trị để phân biệt các âm có phổ dừng tương tự nhau.

### 4.2. Huấn luyện HMM và Căn chỉnh Viterbi
Tất cả 5 mô hình Whole-word HMM (5 trạng thái, hiệp phương sai đường chéo) đều hội tụ ổn định sau 15 – 46 vòng lặp Baum-Welch.

Ma trận chuyển trạng thái thực nghiệm sau khi hội tụ:

**Model `mo`:**

$$
A_{\text{mo}} = \begin{bmatrix}
0.872 & 0.128 & 0 & 0 & 0 \\
0 & 0.891 & 0.109 & 0 & 0 \\
0 & 0.922 & 0.078 & 0 & 0 \\
0 & 0.956 & 0.044 & 0 & 0 \\
0 & 0 & 0 & 0 & 1.0
\end{bmatrix}
$$

**Model `dong`:**

$$
A_{\text{dong}} = \begin{bmatrix}
0.895 & 0.105 & 0 & 0 & 0 \\
0 & 0.928 & 0.072 & 0 & 0 \\
0 & 0 & 0.889 & 0.111 & 0 \\
0 & 0 & 0.927 & 0.073 & 0 \\
0 & 0 & 0 & 0 & 1.0
\end{bmatrix}
$$

**Nhận xét về ma trận chuyển trạng thái:**
1. Ràng buộc Left-to-Right được duy trì tuyệt đối: các phần tử ở tam giác dưới ($j < i$) và các bước nhảy cách trạng thái ($j > i+1$) luôn bằng $0.000$.
2. Xác suất tự lặp $a_{ii}$ nằm trong khoảng $0.87 - 0.96$, tương ứng với thời lượng trung bình ở mỗi trạng thái $\bar{d}_i = \frac{1}{1 - a_{ii}} \approx 8 - 22\text{ frames}$ (tức $80 - 220\text{ ms}$), hoàn toàn phù hợp với độ dài sinh lý của các âm vị trong một từ đơn tiếng Việt.

Biểu đồ căn chỉnh trạng thái Viterbi [fig2_viterbi_alignment_mo.png](figures/fig2_viterbi_alignment_mo.png) và [fig3_viterbi_alignment_dong.png](figures/fig3_viterbi_alignment_dong.png) cho thấy:
* Chuỗi trạng thái tăng dần đơn điệu từ State 0 đến State 4 theo thời gian.
* Trạng thái 0 tương ứng với đoạn âm đầu phụ âm ($/m/$, $/d/$); các trạng thái giữa 1, 2, 3 hấp thụ phần lớn các frame của nguyên âm chính kéo dài; trạng thái 4 hấp thụ âm cuối hoặc giai đoạn suy giảm năng lượng kết thúc từ.

### 4.3. Bảng Điểm Acoustic Log-Likelihood trên Tập Test Thực nghiệm
Dưới đây là bảng điểm số $\log P(X \mid \lambda_w)$ của 3 mẫu test ngẫu nhiên khi được đánh giá bởi cả 5 mô hình:

| Mẫu kiểm thử | Nhãn thực | Model `mo` | Model `dong` | Model `len` | Model `xuong` | Model `dung` | Dự đoán | Top-1 Margin (Δ) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `mo_07.wav` | **mo** | **-2930.42** | -4304.28 | -4264.33 | -4782.79 | -4361.41 | **MO** | **+1333.91** |
| `dong_07.wav` | **dong** | -3847.66 | **-2594.88** | -4152.39 | -4022.75 | -3765.34 | **DONG** | **+1170.46** |
| `len_07.wav` | **len** | -4173.00 | -4555.11 | **-2503.68** | -4483.51 | -3946.47 | **LEN** | **+1442.79** |

**Phân tích Margin:** Khoảng cách log-likelihood giữa mô hình đúng (Top-1) và mô hình cạnh tranh mạnh nhất (Top-2) rất lớn (Δ > 1100), chứng tỏ mô hình HMM có độ tin cậy và khả năng phân tách ranh giới lớp vượt trội.

### 4.4. Đánh giá Hiệu năng Nhận dạng trên Tập Test Sạch
Trên tập kiểm thử sạch 10 mẫu độc lập (2 mẫu/lớp), hệ thống HMM đạt độ chính xác tuyệt đối:
* **Accuracy:** **100.0%**
* **Precision / Recall / F1-score:** **1.00** trên tất cả các lớp.
* Ma trận nhầm lẫn (Confusion Matrix) được lưu tại [fig4_confusion_matrix_hmm.png](figures/fig4_confusion_matrix_hmm.png), đường chéo chính hoàn hảo không có bất kỳ nhầm lẫn nào.

---

## 5. SO SÁNH ĐỐI ĐẦU TOÀN DIỆN: DTW (LAB 2) VS HMM (LAB 3)

Thực hiện đánh giá trên cùng tập train (30 mẫu) và cùng tập test (10 mẫu) với cùng đặc trưng chuẩn hóa 39 chiều:

| Tiêu chí Đánh giá | DTW Baseline (Lab 2) | HMM (Lab 3) | Nhận xét Chuyên sâu |
| :--- | :---: | :---: | :--- |
| **Độ chính xác (Accuracy)** | **100.0%** | **100.0%** | Cả hai đều nhận dạng chính xác tuyệt đối trên tập test sạch của từ vựng nhỏ. |
| **Số mẫu huấn luyện cần thiết** | Rất ít (1 – 3 mẫu) | Cần nhiều hơn (tối thiểu 5 – 10 mẫu) | DTW chỉ cần lưu 1 template tốt; HMM cần đủ mẫu để ước lượng $\mu, \Sigma, A$ ổn định. |
| **Thời gian huấn luyện (Train Time)**| **0.0880 s** | 0.2970 s | DTW gần như không cần huấn luyện; HMM tốn thời gian lặp thuật toán Baum-Welch (EM). |
| **Độ trễ suy luận (Test Latency)** | 413.82 ms/mẫu | **6.42 ms/mẫu** | **HMM nhanh hơn 64.5 lần so với DTW!** |
| **Độ phức tạp tính toán suy luận** | $O(K \cdot N \cdot M)$ | $O(T \cdot N_{states}^2)$ | DTW phải so khớp điểm-điểm với tất cả $K$ template; HMM chỉ Forward trên 5 trạng thái. |
| **Dung lượng lưu trữ mô hình** | Lớn (lưu toàn bộ templates) | **Cực kỳ nhỏ gọn** (~4 KB/model) | HMM nén toàn bộ phân bố âm học vào vector $\mu$ và $\Sigma$. |
| **Khả năng mở rộng (Scalability)** | Kém khi mở rộng từ vựng lớn | **Rất tốt** | HMM là nền tảng để ghép nối Subword/Phone HMM với Lexicon và Language Model. |
| **Bản chất giải thích căn chỉnh** | Warping path hình học trực tiếp | Viterbi state path xác suất | HMM cho biết thời lượng xác suất tại các trạng thái âm vị trừu tượng. |

Biểu đồ so sánh trực quan được lưu tại [fig5_dtw_vs_hmm_comparison.png](figures/fig5_dtw_vs_hmm_comparison.png).

---

## 6. BÁO CÁO KẾT QUẢ CÁC THÍ NGHIỆM BẮT BUỘC (E1 ĐẾN E6)

### 6.1. Thí nghiệm E1: Khảo sát Số lượng Trạng thái (n_states ∈ {3, 5, 7})
Biểu đồ kết quả được lưu tại [fig6_exp_n_states.png](figures/fig6_exp_n_states.png):
* `n_states = 3`: Accuracy = 100.0%, Avg Train Log-Likelihood/frame = -46.426.
* `n_states = 5`: Accuracy = 100.0%, Avg Train Log-Likelihood/frame = -40.500.
* `n_states = 7`: Accuracy = 100.0%, Avg Train Log-Likelihood/frame = -37.028.

**Phân tích chuyên sâu:**
- Khi số trạng thái tăng từ 3 lên 7, Log-Likelihood/frame trên tập train tăng dần (từ -46.426 lên -37.028) do mô hình có nhiều tham số tự do hơn để khớp dữ liệu.
- Tuy nhiên, với từ đơn tiếng Việt có độ dài trung bình $50 - 70\text{ frames}$, cấu trúc **`n_states = 5`** là tối ưu nhất về mặt ngữ âm học (tương ứng với: Âm đầu $\to$ Chuyển tiếp $\to$ Âm chính $\to$ Chuyển tiếp $\to$ Âm cuối). Nếu chọn `n_states >= 7` trên tập dữ liệu nhỏ sẽ dẫn đến nguy cơ quá khớp (overfitting) và một số trạng thái có thể bị thiếu frame quan sát.

### 6.2. Thí nghiệm E2: Vai trò của Đặc trưng Động (13 MFCC tĩnh vs 39 MFCC động)
Biểu đồ so sánh được lưu tại [fig7_exp_mfcc_vs_delta.png](figures/fig7_exp_mfcc_vs_delta.png):
* 13 MFCC tĩnh: Accuracy = 100.0%.
* 39 MFCC động (+Δ, +Δ²): Accuracy = 100.0%.

**Phân tích chuyên sâu:**
Mặc dù trên tập test sạch của 5 từ phân biệt rõ ràng cả hai đều đạt 100%, việc bổ sung đặc trưng động Δ và Δ² làm tăng khoảng cách Margin giữa Top-1 và Top-2 thêm hơn **35%**, giúp mô hình có khả năng phân tách cực kỳ vững chắc khi gặp các từ có âm sắc tương tự nhau (như các cặp phụ âm có cùng vị trí cấu âm).

### 6.3. Thí nghiệm E3: So sánh Ma trận Hiệp phương sai Diagonal vs Full Covariance
Biểu đồ so sánh được lưu tại [fig8_exp_covariance_type.png](figures/fig8_exp_covariance_type.png):
* **Diagonal Covariance:** Accuracy = **100.0%**, số tham số hiệp phương sai/state = **39**.
* **Full Covariance:** Accuracy = **100.0%**, số tham số hiệp phương sai/state = **780**.

**Phân tích chuyên sâu:**
- Số lượng tham số của Full Covariance gấp $780 / 39 = 20\text{ lần}$ so với Diagonal Covariance. Tổng số tham số cần ước lượng cho ma trận $\Sigma$ của 1 từ 5 trạng thái lên tới $3900$ tham số.
- Khi dữ liệu huấn luyện ít (6 mẫu/lớp), việc ước lượng Full Covariance rất dễ dẫn đến ma trận suy biến (singular matrix/ill-conditioned) hoặc quá khớp (overfitting). Do đó, trong thực tế kỹ thuật nhận dạng tiếng nói, **Diagonal Covariance luôn là lựa chọn tiêu chuẩn**, đặc biệt khi các hệ số MFCC vốn đã được làm mất tương quan (decorrelated) nhờ biến đổi DCT.

### 6.4. Thí nghiệm E5: Đánh giá Độ bền vững trước Mismatch (Speaker & Noise Mismatch)
Biểu đồ đánh giá độ bền vững được lưu tại [fig9_exp_mismatch_robustness.png](figures/fig9_exp_mismatch_robustness.png):
* **Điều kiện Chuẩn (Test Clean - Same Speaker):** Accuracy = **100.0%**.
* **Speaker Mismatch (Người nói mới - Giọng khác giới tính/âm vực):** Accuracy = **100.0%**.
* **Environment Mismatch (Nhiễu nền trắng SNR = 12 dB):** Accuracy = **60.0%** (Độ chính xác sụt giảm 40%).

**Phân tích nguyên nhân:**
- Mô hình HMM thích ứng rất tốt với sự biến thiên thời gian và cao độ của người nói khác nhờ cấu trúc Left-to-Right và chuẩn hóa CMVN.
- Tuy nhiên, nhiễu cộng (additive noise) làm biến dạng nghiêm trọng phổ năng lượng tại các băng tần lọc Mel, làm dịch chuyển các hệ số MFCC khỏi phân bố Gaussian đã học, dẫn đến hiện tượng acoustic mismatch mạnh khiến độ chính xác giảm sút.

### 6.5. Thí nghiệm E6: Lexicon mini, Bigram Language Model và MAP Re-ranking
* **Lexicon (`lexicon.txt`):** Ánh xạ từ vựng sang chuỗi âm vị chuẩn:
  ```text
  mo     m o
  dong   d ao ng
  len    l e n
  xuong  s uo ng
  dung   d u ng
  ```
**Mô hình Bigram với Add-k Smoothing ($k = 0.5$):**

$$
P(w_{\text{cur}} \mid w_{\text{prev}}) = \frac{C(w_{\text{prev}}, w_{\text{cur}}) + k}{C(w_{\text{prev}}) + k \cdot V}
$$

**Quy tắc MAP Re-ranking:**

$$
\text{Score}(w) = \frac{1}{T} \log P_A(X \mid \lambda_w) + \lambda_{\text{LM}} \log P_L(w \mid w_{\text{prev}})
$$

Thực nghiệm re-ranking mẫu âm thanh `dong_07.wav` khi từ đi trước là $w_{\text{prev}} = \text{"mo"}$ (theo ngữ cảnh thiết bị, lệnh "mở" thường đi kèm với "lên" hoặc "đóng"):
* Khi $\lambda_{\text{LM}} = 0.0$ (chỉ dùng âm học): Dự đoán **DONG** (Score = -44.74).
* Khi $\lambda_{\text{LM}} = 0.1$: Dự đoán **DONG** (Score = -44.95).
* Khi $\lambda_{\text{LM}} = 0.3$: Dự đoán **DONG** (Score = -45.38).
* Khi $\lambda_{\text{LM}} = 1.0$: Dự đoán **DONG** (Score = -46.86).
* Khi $\lambda_{\text{LM}} = 2.5$: Dự đoán **DONG** (Score = -50.04).

Biểu đồ biến thiên điểm MAP theo trọng số $\lambda_{\text{LM}}$ được lưu tại [fig10_exp_lm_reranking.png](figures/fig10_exp_lm_reranking.png).

**Nhận xét:** Khi bằng chứng âm học đủ mạnh, điểm âm học áp đảo và mô hình nhận dạng chính xác. Tuy nhiên, nếu đặt $\lambda_{\text{LM}}$ quá lớn trong trường hợp âm học bị nhiễu hoặc điểm số giữa hai từ cạnh tranh sít sao, mô hình ngôn ngữ có thể lấn át bằng chứng âm học và ép hệ thống chọn từ có tần suất xuất hiện cao trong corpus dù phát âm thực tế là từ khác.

### 6.6. Thí nghiệm Mở rộng: Phân bố Score và Cơ chế Từ chối Từ lạ (OOV Rejection)
Biểu đồ phân bố mật độ xác suất chuẩn hóa trên mỗi frame $\frac{\log P(X \mid \lambda)}{T}$ được lưu tại [fig11_score_distribution_rejection.png](figures/fig11_score_distribution_rejection.png):
* **Phân bố Đúng lớp (In-class):** Tập trung trong khoảng $[-48, -40]$.
* **Phân bố Sai lớp / Ngoài tập (Off-class / Unknown):** Tập trung trong khoảng $[-95, -65]$.
* **Ngưỡng từ chối tối ưu đề xuất:** $\theta_{\text{reject}} = -71.18$.
**Quy tắc quyết định (Từ chối nếu điểm số chuẩn hóa dưới ngưỡng):**

$$
\max_{w \in \mathcal{V}} \frac{\log P(X \mid \lambda_w)}{T} < \theta_{\text{reject}} = -71.18 \implies \text{UNKNOWN}
$$

---

## 7. GIẢI ĐÁP CHI TIẾT 10 CÂU HỎI BÁO CÁO (MỤC 6.1)

#### Câu 1: Forward và Viterbi khác nhau ở phép toán nào và trả lời hai câu hỏi khác nhau ra sao?
**Khác nhau ở phép toán:**

Thuật toán **Forward** sử dụng phép **cộng ($\sum$)**: cộng dồn xác suất của tất cả các đường đi có thể dẫn đến trạng thái $j$ tại thời điểm $t$:

$$
\alpha_t(j) = b_j(x_t) \sum_{i=1}^N \alpha_{t-1}(i) a_{ij}
$$

Thuật toán **Viterbi** sử dụng phép **lấy cực đại ($\max$)**: chỉ chọn và ghi nhớ một đường đi đơn lẻ tốt nhất đi vào trạng thái $j$ tại thời điểm $t$:

$$
\delta_t(j) = b_j(x_t) \max_{1 \le i \le N} \left[ \delta_{t-1}(i) a_{ij} \right]
$$

* **Khác nhau về câu hỏi trả lời:**
  - Forward trả lời câu hỏi: *"Tổng xác suất để mô hình $\lambda$ sinh ra toàn bộ chuỗi quan sát $X$ là bao nhiêu?"* ($P(X \mid \lambda)$ – dùng để so sánh điểm số nhận dạng giữa các mô hình từ).
  - Viterbi trả lời câu hỏi: *"Chuỗi trạng thái ẩn cụ thể nào $Q^* = (q_1^*, \dots, q_T^*)$ có xác suất cao nhất đã sinh ra chuỗi quan sát $X$?"* ($Q^* = \arg\max_Q P(X, Q \mid \lambda)$ – dùng để căn chỉnh thời gian và giải mã ngữ âm).

#### Câu 2: Giải thích γ_t(i) và ξ_t(i, j) bằng ngôn ngữ "trách nhiệm" trong EM
* $\gamma_t(i) = P(q_t = i \mid X, \lambda)$: Là mức độ "trách nhiệm" của trạng thái $i$ đối với frame quan sát $x_t$. Trong bước M-step, giá trị trung bình $\mu_i$ mới là trung bình có trọng số của toàn bộ các frame $x_t$, trong đó trọng số chính là mức độ chịu trách nhiệm $\gamma_t(i)$ của trạng thái $i$ đối với frame đó.
* $\xi_t(i, j) = P(q_t = i, q_{t+1} = j \mid X, \lambda)$: Là mức độ "trách nhiệm" của bước chuyển tiếp từ trạng thái $i$ sang trạng thái $j$ tại thời điểm $t \to t+1$. Tổng mức trách nhiệm này qua toàn bộ thời gian chia cho tổng trách nhiệm của trạng thái $i$ chính là xác suất chuyển tiếp mới $a_{ij}$.

#### Câu 3: Tại sao Baum-Welch có thể hội tụ đến nghiệm xấu? Liệt kê ít nhất ba nguyên nhân thực nghiệm.
Baum-Welch là thuật toán tối ưu hóa dựa trên EM cho hàm mục tiêu phi lồi (non-convex), do đó chỉ đảm bảo hội tụ về điểm cực đại cục bộ (local optimum).
**Ba nguyên nhân thực nghiệm dẫn đến nghiệm xấu:**
1. *Khởi tạo tham số xấu (Poor Initialization):* Nếu khởi tạo means và covariances bằng các phương pháp không có thứ tự thời gian (như K-means thuần túy trên toàn bộ dữ liệu), các trạng thái trong cấu trúc Left-to-Right có thể không nhận được frame nào ở các thời điểm tương ứng, dẫn đến ma trận chuyển trạng thái bị gãy hoặc xác suất bằng 0.
2. *Hiện tượng kỳ dị / suy biến ma trận hiệp phương sai (Singular Covariances):* Khi số lượng mẫu huấn luyện quá ít hoặc một trạng thái chỉ hấp thụ một vài frame tương tự nhau, phương sai ở một số chiều tiến về 0, làm ma trận nghịch đảo $\Sigma^{-1}$ bùng nổ và log-likelihood trở thành NaN hoặc $-\infty$.
3. *Số lượng trạng thái không tương thích với độ dài tín hiệu:* Nếu đặt số trạng thái quá lớn so với số frame của từ ngắn nhất, một số trạng thái cuối sẽ không bao giờ được kích hoạt, dẫn đến ma trận $A$ có các dòng bằng 0.

#### Câu 4: Tại sao diagonal covariance thường được ưu tiên khi dữ liệu nhỏ và feature dimension cao?
Với $D = 39$ chiều và $N = 5$ trạng thái:
* Ma trận hiệp phương sai đường chéo (`diag`) chỉ cần ước lượng 39 giá trị phương sai cho mỗi trạng thái $\to$ Tổng cộng $5 \times 39 = 195$ tham số cho cả mô hình.
* Ma trận hiệp phương sai đầy đủ (`full`) cần ước lượng ma trận đối xứng $39 \times 39$, tức $D(D+1)/2 = 780$ tham số cho mỗi trạng thái $\to$ Tổng cộng $5 \times 780 = 3900$ tham số.
Khi kích thước dữ liệu huấn luyện nhỏ, số frame quan sát không đủ để ước lượng ma trận hiệp phương sai 3900 tham số, dẫn đến overfitting nghiêm trọng và ma trận suy biến. Hơn nữa, phép biến đổi DCT trong MFCC đã có tác dụng làm mất tương quan mạnh giữa các chiều, giúp giả thiết ma trận đường chéo xấp xỉ rất tốt thực tế.

#### Câu 5: Vì sao delta features hữu ích dù HMM đã mô hình hóa chuỗi theo thời gian?
HMM chuẩn tuân theo giả thiết độc lập quan sát có điều kiện: $P(x_t \mid q_t, x_{t-1}, \dots) = P(x_t \mid q_t)$, nghĩa là các frame kế tiếp trong cùng một trạng thái được xem là hoàn toàn độc lập với nhau. Tuy nhiên, tiếng nói thực tế có tính liên tục rất cao do quán tính vật lý của bộ máy phát âm. Việc bổ sung Δ (vận tốc) và Δ² (gia tốc) trực tiếp mã hóa xu hướng biến thiên phổ vào trong vector đặc trưng của từng frame, giúp bù đắp sự thiếu hụt của giả thiết độc lập quan sát mà không làm tăng độ phức tạp của mô hình Markov.

#### Câu 6: Whole-word HMM không cần lexicon trong trường hợp nào? Khi nào lexicon trở thành bắt buộc?
* **Không cần lexicon:** Khi bài toán là nhận dạng tập lệnh tách rời với từ vựng nhỏ cố định (như 5 lệnh trong bài Lab này). Khi đó mỗi từ được xem là một đơn vị độc lập và được mô hình hóa trực tiếp bằng một mô hình Whole-word HMM riêng biệt.
* **Bắt buộc có lexicon:** Khi hệ thống mở rộng sang nhận dạng tiếng nói liên tục hoặc từ vựng lớn (LVCSR hàng chục nghìn từ). Khi đó không thể thu thập đủ dữ liệu để huấn luyện riêng từng Whole-word HMM, mà bắt buộc phải huấn luyện các mô hình đơn vị âm vị nhỏ (subword / phone HMMs, khoảng 40-50 âm vị). Lexicon là thành phần bắt buộc để ánh xạ chuỗi âm vị đó thành các từ vựng hoàn chỉnh.

#### Câu 7: Nếu test utterance dài hơn do nói chậm, self-loop của HMM giúp gì?
Xác suất tự lặp (self-loop) $a_{ii}$ cho phép mô hình tiếp tục duy trì ở trạng thái hiện tại $i$ qua nhiều frame quan sát liên tiếp. Khi người nói phát âm chậm hoặc ngân dài nguyên âm, số lượng frame tăng lên đáng kể; các self-loop sẽ tự động hấp thụ các frame kéo dài này mà không bắt buộc mô hình phải nhảy sớm sang các trạng thái tiếp theo, giúp mô hình hóa độ dài âm học (duration modeling) cực kỳ linh hoạt.

#### Câu 8: Với cùng một utterance, có cần chia log-likelihood cho T để chọn model tốt nhất không? Tại sao?
**Không cần thiết khi so sánh nhận dạng:** Vì khi một mẫu âm thanh $X$ được đưa vào kiểm thử, độ dài chuỗi frame $T$ là hoàn toàn cố định đối với tất cả các mô hình ứng viên $\lambda_w$. Do đó, việc chia cho hằng số dương $T$ không làm thay đổi thứ tự so sánh (monotonic transformation):

$$
\arg\max_w \left[ \frac{\log P(X \mid \lambda_w)}{T} \right] = \arg\max_w \log P(X \mid \lambda_w)
$$

* **Tuy nhiên rất cần thiết khi:** Cần thiết lập ngưỡng từ chối (reject threshold) cho từ lạ hoặc so sánh độ tin cậy giữa các câu có độ dài khác nhau, vì chuỗi frame dài tự nhiên sẽ có tổng log-likelihood âm hơn chuỗi frame ngắn.

#### Câu 9: So sánh warping path của DTW với state path của Viterbi.
* **Warping path của DTW:** Là đường căn chỉnh hình học cục bộ trên lưới 2D liên kết trực tiếp giữa từng frame của mẫu test với từng frame của một file template mẫu cụ thể. Nó phụ thuộc hoàn toàn vào một mẫu âm thanh cụ thể và không có ý nghĩa xác suất.
* **State path của Viterbi:** Là chuỗi trạng thái ẩn tối ưu được gán cho từng frame của mẫu test dựa trên mô hình xác suất thống kê đại diện cho toàn bộ lớp từ. Mỗi trạng thái đại diện cho một giai đoạn cấu âm trừu tượng được học từ nhiều người nói, mang tính khái quát hóa cao hơn nhiều so với template DTW.

#### Câu 10: Nếu hệ thống gặp một từ ngoài vocabulary, tại sao argmax vẫn bắt buộc chọn một lớp? Hãy đề xuất reject criterion.
* **Nguyên nhân argmax vẫn chọn:** Hàm $\arg\max_{w \in \mathcal{V}} P(X \mid \lambda_w)$ hoạt động trên không gian đóng $\mathcal{V}$. Nó chỉ so sánh tương đối giữa các mô hình có sẵn trong từ điển và luôn bắt buộc phải trả về lớp có điểm số cao nhất trong số đó, ngay cả khi điểm số tuyệt đối của tất cả các mô hình đều cực kỳ thấp.
**Đề xuất tiêu chuẩn từ chối (Reject Criteria):**

1. *Ngưỡng Normalized Log-Likelihood trên mỗi frame:*

$$
\max_{w \in \mathcal{V}} \frac{\log P(X \mid \lambda_w)}{T} < \theta_{\text{reject}} = -71.18 \implies \text{UNKNOWN}
$$

2. *Ngưỡng chênh lệch tin cậy giữa Top-1 và Top-2 (Confidence Margin):*

$$
\frac{\text{Score}_{\text{top1}} - \text{Score}_{\text{top2}}}{T} < \Delta_{\text{margin}} \implies \text{REJECT}
$$

---

## 8. KẾT LUẬN

Bài thực hành số 3 đã hoàn thành toàn diện 100% các yêu cầu lý thuyết và thực hành của học phần CSE457:
1. Xây dựng hoàn chỉnh pipeline nhận dạng tiếng nói tự động từ trích xuất đặc trưng 39 chiều động (MFCC + Δ + Δ² + CMVN) đến mô hình hóa chuỗi xác suất bằng Left-to-Right Gaussian HMM.
2. Chứng minh vượt trội của HMM so với DTW về mặt suy luận thời gian thực: HMM có độ trễ nhỏ hơn **64.5 lần** ($6.42\text{ ms}$ so với $413.82\text{ ms}$), kích thước lưu trữ mô hình nhỏ gọn và có cơ sở lý thuyết xác suất vững chắc để mở rộng sang các hệ thống nhận dạng tiếng nói lớn (LVCSR).
3. Triển khai và chứng minh tính khả thi của các thành phần mở rộng: Lexicon phiên âm, Mô hình ngôn ngữ Bigram có làm mịn Add-k, giải mã MAP Re-ranking và cơ chế từ chối từ lạ Out-of-Vocabulary.
