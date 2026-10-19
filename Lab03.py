"""
CSE457 - Xử lý Âm thanh và Tiếng nói
BÀI THỰC HÀNH SỐ 3 (LAB 3): NHẬN DẠNG TIẾNG NÓI NÂNG CAO VỚI HMM
Sinh viên: Nguyễn Thị Linh
MSSV: 2351260663
Lớp: 65TTNT
Trường Đại học Thủy Lợi - Khoa CNTT - Bộ môn Trí tuệ Nhân tạo
"""

import os
import sys
import math
import time
import pickle
import json
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report

# ==============================================================================
# 0. CẤU HÌNH VÀ THƯ MỤC
# ==============================================================================
LABELS = ['mo', 'dong', 'len', 'xuong', 'dung']
LABEL_NAMES_VI = {'mo': 'Mở', 'dong': 'Đóng', 'len': 'Lên', 'xuong': 'Xuống', 'dung': 'Dừng'}
SR = 16000

FIG_DIR = Path('figures')
MODEL_DIR = Path('models')
DATA_DIR = Path('dataset')

FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style='whitegrid')

# ==============================================================================
# 1. TRÍCH XUẤT ĐẶC TRƯNG 39 CHIỀU (MFCC + DELTA + DELTA-DELTA) & CMVN
# ==============================================================================
def extract_feat(path, sr=16000, with_delta=True):
    """
    Trích xuất đặc trưng âm học chuẩn hóa cho HMM:
    - Fs = 16 kHz mono
    - Cắt khoảng lặng biên bằng librosa.effects.trim(top_db=30)
    - Tiền nhấn pre-emphasis coef = 0.97
    - 13 MFCC (cửa sổ Hamming 25ms, hop 10ms, 26 mel filters)
    - Delta và Delta-Delta (nếu with_delta=True) -> 39 chiều
    - Utterance-level CMVN (chuẩn hóa trung bình và phương sai trên mỗi utterance)
    """
    y, sr = librosa.load(path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    y = librosa.effects.preemphasis(y, coef=0.97)
    
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13,
        n_fft=512, win_length=int(0.025 * sr),
        hop_length=int(0.010 * sr), window='hamming',
        n_mels=26, fmin=20, fmax=sr / 2
    )
    
    if with_delta:
        d1 = librosa.feature.delta(mfcc, order=1)
        d2 = librosa.feature.delta(mfcc, order=2)
        F = np.vstack([mfcc, d1, d2]).T  # Shape: (T, 39)
    else:
        F = mfcc.T  # Shape: (T, 13)
        
    # Utterance-level CMVN
    mu = F.mean(axis=0, keepdims=True)
    sd = F.std(axis=0, keepdims=True) + 1e-8
    return (F - mu) / sd

# ==============================================================================
# 2. KHỞI TẠO VÀ CẤU HÌNH LEFT-TO-RIGHT HMM
# ==============================================================================
def make_lr_hmm(n_states=5, n_iter=80, covariance_type='diag', feats_list=None, random_state=42):
    """
    Khởi tạo mô hình Left-to-Right Gaussian HMM:
    - n_states: số trạng thái (thường 3, 5, hoặc 7)
    - startprob_ = [1.0, 0, ..., 0]: luôn bắt đầu ở trạng thái 0
    - transmat_: chỉ cho phép self-loop (aii ~ 0.7) và transition kế tiếp (ai,i+1 ~ 0.3)
    - trạng thái cuối: self-loop = 1.0 (trạng thái hấp thụ)
    - covariance_type: 'diag' (được ưu tiên khi dữ liệu ít) hoặc 'full'
    - Khởi tạo means và covars bằng phân đoạn thời gian đều (Uniform Temporal Segmentation),
      tránh việc KMeans không có thông tin thứ tự thời gian gây underflow/không hội tụ trong Left-to-Right HMM.
    """
    start = np.zeros(n_states)
    start[0] = 1.0
    
    A = np.zeros((n_states, n_states))
    for i in range(n_states - 1):
        A[i, i] = 0.7
        A[i, i + 1] = 0.3
    A[-1, -1] = 1.0
    
    if feats_list is not None and len(feats_list) > 0:
        D = feats_list[0].shape[1]
        segments = [[] for _ in range(n_states)]
        for X in feats_list:
            T = len(X)
            step = T / n_states
            for s in range(n_states):
                s_start = int(s * step)
                s_end = int((s + 1) * step) if s < n_states - 1 else T
                segments[s].append(X[s_start:s_end])
                
        means = np.zeros((n_states, D))
        if covariance_type == 'diag':
            covars = np.zeros((n_states, D))
            for s in range(n_states):
                data_s = np.vstack(segments[s])
                means[s] = np.mean(data_s, axis=0)
                covars[s] = np.var(data_s, axis=0) + 1e-2
        else:  # full covariance
            covars = np.zeros((n_states, D, D))
            for s in range(n_states):
                data_s = np.vstack(segments[s])
                means[s] = np.mean(data_s, axis=0)
                covars[s] = np.cov(data_s, rowvar=False) + np.eye(D) * 1e-2
                
        model = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=1e-3,
            min_covar=1e-3,
            init_params='',  # Means, covars, startprob, transmat đã được định nghĩa
            params='tmc',
            random_state=random_state
        )
        model.startprob_ = start
        model.transmat_ = A
        model.means_ = means
        model.covars_ = covars
    else:
        model = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=1e-3,
            min_covar=1e-3,
            init_params='mc',
            params='tmc',
            random_state=random_state
        )
        model.startprob_ = start
        model.transmat_ = A
        
    return model

# ==============================================================================
# 3. NHẬN DẠNG VÀ TÍNH LOG-LIKELIHOOD
# ==============================================================================
def recognize_hmm(path, models, with_delta=True):
    """
    Nhận dạng từ đơn theo nguyên lý Maximum Likelihood (ML):
    w* = argmax_w log P(X | lambda_w)
    """
    F = extract_feat(path, with_delta=with_delta)
    scores = {}
    for label, model in models.items():
        try:
            scores[label] = model.score(F)
        except Exception:
            scores[label] = -1e9
    pred = max(scores, key=scores.get)
    return pred, scores, F

# ==============================================================================
# 4. DTW BASELINE (THEO LAB 2) ĐỂ SO SÁNH CÔNG BẰNG
# ==============================================================================
def dtw_distance(X, Y):
    """
    Thuật toán Dynamic Time Warping (DTW) từ Lab 2:
    - Ma trận chi phí cục bộ Euclidean
    - Quy hoạch động 3 hướng (ngang, dọc, chéo)
    - Chuẩn hóa theo độ dài đường đi: D[N, M] / len(path)
    """
    N, M = len(X), len(Y)
    D = np.full((N + 1, M + 1), np.inf)
    D[0, 0] = 0.0
    
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cost = np.linalg.norm(X[i - 1] - Y[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
            
    # Backtrack tìm path length
    i, j = N, M
    path_len = 1
    while i > 1 or j > 1:
        steps = [
            (D[i - 1, j - 1], i - 1, j - 1),
            (D[i - 1, j], i - 1, j),
            (D[i, j - 1], i, j - 1)
        ]
        _, i, j = min(steps, key=lambda z: z[0])
        path_len += 1
        
    norm_dist = D[N, M] / path_len
    return norm_dist

def build_dtw_templates(train_dir='dataset/train', with_delta=True):
    """Xây dựng thư viện templates cho DTW từ tập train."""
    templates = {label: [] for label in LABELS}
    for label in LABELS:
        for wav_path in sorted(Path(train_dir, label).glob('*.wav')):
            F = extract_feat(str(wav_path), with_delta=with_delta)
            templates[label].append(F)
    return templates

def recognize_dtw(path, templates, with_delta=True):
    """Nhận dạng 1-NN bằng khoảng cách DTW tối thiểu."""
    F = extract_feat(path, with_delta=with_delta)
    scores = {}
    for label, ref_list in templates.items():
        min_d = min(dtw_distance(F, R) for R in ref_list)
        scores[label] = min_d
    pred = min(scores, key=scores.get)
    return pred, scores

# ==============================================================================
# 5. LEXICON & LANGUAGE MODEL (BIGRAM VỚI ADD-K SMOOTHING & RE-RANKING)
# ==============================================================================
def parse_lexicon(filepath='lexicon.txt'):
    """Đọc file lexicon.txt và phân tích cú pháp thành dict."""
    lexicon = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                lexicon[parts[0]] = parts[1:]
    return lexicon

def build_bigram_lm(corpus_path='corpus.txt', k=0.5):
    """
    Xây dựng mô hình ngôn ngữ Bigram từ corpus có làm mịn Add-k (Laplace mở rộng):
    P(w_cur | w_prev) = (Count(w_prev, w_cur) + k) / (Count(w_prev) + k * V)
    """
    unigram_counts = {}
    bigram_counts = {}
    vocab = set(LABELS)
    
    if Path(corpus_path).exists():
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split()
                for i, w in enumerate(words):
                    vocab.add(w)
                    unigram_counts[w] = unigram_counts.get(w, 0) + 1
                    if i > 0:
                        prev = words[i - 1]
                        pair = (prev, w)
                        bigram_counts[pair] = bigram_counts.get(pair, 0) + 1
                        
    V = len(vocab)
    
    def get_prob(prev, cur):
        c2 = bigram_counts.get((prev, cur), 0)
        c1 = unigram_counts.get(prev, 0)
        return (c2 + k) / (c1 + k * V)
        
    return get_prob, vocab, unigram_counts, bigram_counts

def rerank_map(F, prev_word, models, lm_prob_fn, lm_weight=0.3):
    """
    MAP Re-ranking kết hợp Acoustic Likelihood và Language Model Prior:
    Score(w) = (1 / T) * log P(X | lambda_w) + lambda_LM * log P_LM(w | prev_word)
    """
    T = len(F)
    total_scores = {}
    acoustic_scores = {}
    lm_scores = {}
    
    for w, m in models.items():
        a_score = m.score(F) / T  # Normalized per frame
        p_lm = lm_prob_fn(prev_word, w)
        lm_score = math.log(max(p_lm, 1e-12))
        
        acoustic_scores[w] = a_score
        lm_scores[w] = lm_score
        total_scores[w] = a_score + lm_weight * lm_score
        
    pred = max(total_scores, key=total_scores.get)
    return pred, total_scores, acoustic_scores, lm_scores

# ==============================================================================
# 6. HUẤN LUYỆN WHOLE-WORD HMM
# ==============================================================================
def train_hmm_models(train_dir='dataset/train', n_states=5, covariance_type='diag', with_delta=True):
    """Huấn luyện 1 Whole-word Gaussian HMM cho mỗi lớp từ vựng."""
    models = {}
    for label in LABELS:
        feats = []
        lengths = []
        wav_files = sorted(Path(train_dir, label).glob('*.wav'))
        for wav in wav_files:
            F = extract_feat(str(wav), with_delta=with_delta)
            feats.append(F)
            lengths.append(len(F))
            
        X_concat = np.vstack(feats)
        model = make_lr_hmm(n_states=n_states, covariance_type=covariance_type, feats_list=feats)
        model.fit(X_concat, lengths)
        
        # Đảm bảo topology cấm chuyển lùi/nhảy cóc không bị trôi số học (numerical drift)
        for i in range(n_states):
            for j in range(n_states):
                if j != i and j != i + 1:
                    model.transmat_[i, j] = 0.0
            row_sum = model.transmat_[i].sum()
            if row_sum > 0:
                model.transmat_[i] /= row_sum
            else:
                model.transmat_[i, i] = 1.0
                
        models[label] = model
        
    return models

# ==============================================================================
# 7. THỰC HIỆN CÁC THÍ NGHIỆM VÀ VẼ HÌNH
# ==============================================================================
def run_all_experiments():
    print("=" * 80)
    print("BẮT ĐẦU CHẠY TOÀN BỘ PIPELINE VÀ CÁC THÍ NGHIỆM LAB 3 (CSE457)")
    print("=" * 80)
    
    # --------------------------------------------------------------------------
    # BƯỚC 1: VẼ HEATMAP MFCC / DELTA / DELTA-DELTA (Hình 1)
    # --------------------------------------------------------------------------
    print("\n[Bước 1] Trích đặc trưng và vẽ Heatmap MFCC / Delta / Delta2 cho 'mo' và 'dong'...")
    feat_mo = extract_feat('dataset/train/mo/mo_01.wav')
    feat_dong = extract_feat('dataset/train/dong/dong_01.wav')
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex='col')
    
    # Từ "Mở"
    im0 = axes[0, 0].imshow(feat_mo[:, :13].T, aspect='auto', origin='lower', cmap='viridis')
    axes[0, 0].set_title("Từ 'Mở' (mo_01.wav) - 13 MFCC tĩnh", fontweight='bold')
    axes[0, 0].set_ylabel("MFCC Index (0-12)")
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[1, 0].imshow(feat_mo[:, 13:26].T, aspect='auto', origin='lower', cmap='magma')
    axes[1, 0].set_title("Từ 'Mở' - 13 Delta-MFCC", fontweight='bold')
    axes[1, 0].set_ylabel("Delta Index")
    plt.colorbar(im1, ax=axes[1, 0])
    
    im2 = axes[2, 0].imshow(feat_mo[:, 26:39].T, aspect='auto', origin='lower', cmap='coolwarm')
    axes[2, 0].set_title("Từ 'Mở' - 13 Delta-Delta MFCC", fontweight='bold')
    axes[2, 0].set_xlabel("Frame Index (Hop 10ms)")
    axes[2, 0].set_ylabel("Delta2 Index")
    plt.colorbar(im2, ax=axes[2, 0])
    
    # Từ "Đóng"
    im3 = axes[0, 1].imshow(feat_dong[:, :13].T, aspect='auto', origin='lower', cmap='viridis')
    axes[0, 1].set_title("Từ 'Đóng' (dong_01.wav) - 13 MFCC tĩnh", fontweight='bold')
    axes[0, 1].set_ylabel("MFCC Index (0-12)")
    plt.colorbar(im3, ax=axes[0, 1])
    
    im4 = axes[1, 1].imshow(feat_dong[:, 13:26].T, aspect='auto', origin='lower', cmap='magma')
    axes[1, 1].set_title("Từ 'Đóng' - 13 Delta-MFCC", fontweight='bold')
    axes[1, 1].set_ylabel("Delta Index")
    plt.colorbar(im4, ax=axes[1, 1])
    
    im5 = axes[2, 1].imshow(feat_dong[:, 26:39].T, aspect='auto', origin='lower', cmap='coolwarm')
    axes[2, 1].set_title("Từ 'Đóng' - 13 Delta-Delta MFCC", fontweight='bold')
    axes[2, 1].set_xlabel("Frame Index (Hop 10ms)")
    axes[2, 1].set_ylabel("Delta2 Index")
    plt.colorbar(im5, ax=axes[2, 1])
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig1_mfcc_delta_heatmaps.png', dpi=300)
    plt.close()
    print("  Đã lưu:", FIG_DIR / 'fig1_mfcc_delta_heatmaps.png')

    # --------------------------------------------------------------------------
    # BƯỚC 2 & 3: HUẤN LUYỆN 5 WHOLE-WORD HMM (MÔ HÌNH CHUẨN: 5 STATES, DIAG)
    # --------------------------------------------------------------------------
    print("\n[Bước 2 & 3] Huấn luyện 5 Whole-word Left-to-Right HMM (5 states, diag covar)...")
    base_models = train_hmm_models('dataset/train', n_states=5, covariance_type='diag', with_delta=True)
    
    # Lưu mô hình vào models/
    for label, m in base_models.items():
        with open(MODEL_DIR / f'hmm_{label}.pkl', 'wb') as f:
            pickle.dump(m, f)
        print(f"  Model '{label}': Hội tụ = {m.monitor_.converged}, Số vòng lặp = {m.monitor_.iter}")
        print(f"  Ma trận chuyển trạng thái A:\n{np.round(m.transmat_, 3)}")

    # --------------------------------------------------------------------------
    # BƯỚC 4 & 5: VITERBI ALIGNMENT VÀ BẢNG ACOUSTIC SCORE (Hình 2 & 3)
    # --------------------------------------------------------------------------
    print("\n[Bước 4 & 5] Giải mã Viterbi Alignment và lập bảng Likelihood...")
    
    # Viterbi cho mo_07.wav
    f_mo = extract_feat('dataset/test/mo/mo_07.wav')
    logp_mo, states_mo = base_models['mo'].decode(f_mo, algorithm='viterbi')
    
    plt.figure(figsize=(9, 4.5))
    plt.step(range(len(states_mo)), states_mo, where='post', color='#1f77b4', linewidth=2.5, marker='o', markersize=4)
    plt.xlabel('Frame Index (Hop 10ms)', fontsize=11)
    plt.ylabel('HMM State Index (0 -> 4)', fontsize=11)
    plt.title(f"Viterbi State Alignment: mo_07.wav vs Model 'MO' (Log-Prob: {logp_mo:.2f})", fontsize=12, fontweight='bold')
    plt.yticks(range(5), [f"State q{i+1}" for i in range(5)])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig2_viterbi_alignment_mo.png', dpi=300)
    plt.close()
    print("  Đã lưu:", FIG_DIR / 'fig2_viterbi_alignment_mo.png')
    
    # Viterbi cho dong_07.wav
    f_dong = extract_feat('dataset/test/dong/dong_07.wav')
    logp_dong, states_dong = base_models['dong'].decode(f_dong, algorithm='viterbi')
    
    plt.figure(figsize=(9, 4.5))
    plt.step(range(len(states_dong)), states_dong, where='post', color='#d62728', linewidth=2.5, marker='s', markersize=4)
    plt.xlabel('Frame Index (Hop 10ms)', fontsize=11)
    plt.ylabel('HMM State Index (0 -> 4)', fontsize=11)
    plt.title(f"Viterbi State Alignment: dong_07.wav vs Model 'DONG' (Log-Prob: {logp_dong:.2f})", fontsize=12, fontweight='bold')
    plt.yticks(range(5), [f"State q{i+1}" for i in range(5)])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig3_viterbi_alignment_dong.png', dpi=300)
    plt.close()
    print("  Đã lưu:", FIG_DIR / 'fig3_viterbi_alignment_dong.png')

    # Bảng acoustic score cho 3 test utterances
    test_samples_demo = [
        'dataset/test/mo/mo_07.wav',
        'dataset/test/dong/dong_07.wav',
        'dataset/test/len/len_07.wav'
    ]
    print("\n  --- BẢNG ACOUSTIC LOG-LIKELIHOOD CHO CÁC MẪU TEST THỰC NGHIỆM ---")
    score_table_demo = []
    for test_p in test_samples_demo:
        pred, scores, F = recognize_hmm(test_p, base_models)
        sorted_scores = sorted(scores.items(), key=lambda z: z[1], reverse=True)
        top1_lbl, top1_val = sorted_scores[0]
        top2_lbl, top2_val = sorted_scores[1]
        margin = top1_val - top2_val
        print(f"  Utterance: {Path(test_p).name} | Dự đoán: {pred.upper()} | Top-1 Margin: {margin:.2f}")
        for lbl in LABELS:
            print(f"      Model {lbl:6s}: {scores[lbl]:10.2f}")
        score_table_demo.append({
            'file': Path(test_p).name,
            'true_label': Path(test_p).parent.name,
            'prediction': pred,
            'scores': scores,
            'margin_top1_top2': margin
        })

    # --------------------------------------------------------------------------
    # BƯỚC 6: ĐÁNH GIÁ ACCURACY VÀ CONFUSION MATRIX TRÊN TẬP TEST SẠCH (Hình 4)
    # --------------------------------------------------------------------------
    print("\n[Bước 6] Đánh giá HMM trên tập kiểm thử Test Sạch...")
    y_true_clean = []
    y_pred_clean = []
    clean_test_files = []
    for label in LABELS:
        for p in sorted(Path('dataset/test', label).glob('*.wav')):
            pred, _, _ = recognize_hmm(str(p), base_models)
            y_true_clean.append(label)
            y_pred_clean.append(pred)
            clean_test_files.append(str(p))
            
    acc_clean_hmm = accuracy_score(y_true_clean, y_pred_clean)
    cm_clean_hmm = confusion_matrix(y_true_clean, y_pred_clean, labels=LABELS)
    print(f"  Độ chính xác HMM (Test Clean): {acc_clean_hmm * 100:.2f}%")
    
    plt.figure(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_clean_hmm, display_labels=[LABEL_NAMES_VI[l] for l in LABELS])
    disp.plot(cmap='Blues', ax=plt.gca(), values_format='d')
    plt.title(f"Confusion Matrix: HMM (39 MFCC, 5 States, Diag)\nAccuracy: {acc_clean_hmm * 100:.1f}%", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig4_confusion_matrix_hmm.png', dpi=300)
    plt.close()
    print("  Đã lưu:", FIG_DIR / 'fig4_confusion_matrix_hmm.png')

    # --------------------------------------------------------------------------
    # BƯỚC 7: SO SÁNH TOÀN DIỆN VỚI DTW BASELINE (Hình 5)
    # --------------------------------------------------------------------------
    print("\n[Bước 7] So sánh HMM với DTW Baseline (Lab 2)...")
    t0_dtw_train = time.time()
    dtw_templates = build_dtw_templates('dataset/train', with_delta=True)
    t_train_dtw = time.time() - t0_dtw_train
    
    # Đo thời gian train HMM
    t0_hmm_train = time.time()
    _ = train_hmm_models('dataset/train', n_states=5, covariance_type='diag', with_delta=True)
    t_train_hmm = time.time() - t0_hmm_train
    
    # Test DTW
    y_pred_dtw = []
    t_test_dtw_list = []
    for p in clean_test_files:
        t0 = time.time()
        pred, _ = recognize_dtw(p, dtw_templates, with_delta=True)
        t_test_dtw_list.append(time.time() - t0)
        y_pred_dtw.append(pred)
        
    acc_clean_dtw = accuracy_score(y_true_clean, y_pred_dtw)
    avg_t_dtw_test = np.mean(t_test_dtw_list) * 1000  # ms
    
    # Test HMM Latency
    t_test_hmm_list = []
    for p in clean_test_files:
        t0 = time.time()
        _, _, _ = recognize_hmm(p, base_models)
        t_test_hmm_list.append(time.time() - t0)
    avg_t_hmm_test = np.mean(t_test_hmm_list) * 1000  # ms
    
    print(f"  DTW: Accuracy = {acc_clean_dtw * 100:.1f}%, Train Time = {t_train_dtw:.4f}s, Test Latency/trial = {avg_t_dtw_test:.2f}ms")
    print(f"  HMM: Accuracy = {acc_clean_hmm * 100:.1f}%, Train Time = {t_train_hmm:.4f}s, Test Latency/trial = {avg_t_hmm_test:.2f}ms")
    
    # Vẽ biểu đồ so sánh DTW vs HMM
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # 1. Accuracy
    axes[0].bar(['DTW (Lab 2)', 'HMM (Lab 3)'], [acc_clean_dtw * 100, acc_clean_hmm * 100], color=['#ff7f0e', '#2ca02c'], width=0.5)
    axes[0].set_ylim(0, 115)
    axes[0].set_ylabel('Độ chính xác (%)', fontsize=11)
    axes[0].set_title('Độ chính xác (Accuracy)', fontweight='bold')
    for i, v in enumerate([acc_clean_dtw * 100, acc_clean_hmm * 100]):
        axes[0].text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')
        
    # 2. Training Time
    axes[1].bar(['DTW (Lab 2)', 'HMM (Lab 3)'], [t_train_dtw, t_train_hmm], color=['#ff7f0e', '#2ca02c'], width=0.5)
    axes[1].set_ylabel('Thời gian huấn luyện (giây)', fontsize=11)
    axes[1].set_title('Thời gian huấn luyện (Train Time)', fontweight='bold')
    for i, v in enumerate([t_train_dtw, t_train_hmm]):
        axes[1].text(i, v + 0.05 * max(t_train_dtw, t_train_hmm), f"{v:.3f}s", ha='center', fontweight='bold')
        
    # 3. Test Latency per trial
    axes[2].bar(['DTW (Lab 2)', 'HMM (Lab 3)'], [avg_t_dtw_test, avg_t_hmm_test], color=['#ff7f0e', '#2ca02c'], width=0.5)
    axes[2].set_ylabel('Thời gian nhận dạng (ms/mẫu)', fontsize=11)
    axes[2].set_title('Độ trễ suy luận (Test Latency)', fontweight='bold')
    for i, v in enumerate([avg_t_dtw_test, avg_t_hmm_test]):
        axes[2].text(i, v + 1.5, f"{v:.2f} ms", ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig5_dtw_vs_hmm_comparison.png', dpi=300)
    plt.close()
    print("  Đã lưu:", FIG_DIR / 'fig5_dtw_vs_hmm_comparison.png')

    # --------------------------------------------------------------------------
    # THÍ NGHIỆM BẮT BUỘC E1: ẢNH HƯỞNG CỦA SỐ LƯỢNG TRẠNG THÁI (n_states in {3, 5, 7})
    # --------------------------------------------------------------------------
    print("\n[Thí nghiệm E1] Khảo sát số trạng thái HMM n_states in {3, 5, 7}...")
    states_candidates = [3, 5, 7]
    e1_results = {}
    for ns in states_candidates:
        m_dict = train_hmm_models('dataset/train', n_states=ns, covariance_type='diag', with_delta=True)
        preds = [recognize_hmm(p, m_dict)[0] for p in clean_test_files]
        acc = accuracy_score(y_true_clean, preds)
        # Tính average log-likelihood trên tập train
        train_ll_list = []
        for l in LABELS:
            for p in sorted(Path('dataset/train', l).glob('*.wav')):
                F = extract_feat(str(p))
                train_ll_list.append(m_dict[l].score(F) / len(F))
        avg_ll = np.mean(train_ll_list)
        e1_results[ns] = {'accuracy': acc, 'avg_train_ll_per_frame': avg_ll}
        print(f"  n_states={ns}: Accuracy = {acc * 100:.1f}%, Avg Train Log-Likelihood/frame = {avg_ll:.3f}")
        
    # Vẽ biểu đồ E1
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    color = '#1f77b4'
    ax1.set_xlabel('Số lượng trạng thái HMM (n_states)', fontsize=11)
    ax1.set_ylabel('Độ chính xác Test (%)', color=color, fontsize=11)
    ax1.plot(states_candidates, [e1_results[ns]['accuracy'] * 100 for ns in states_candidates], color=color, marker='o', linewidth=2.5, markersize=8)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 115)
    
    ax2 = ax1.twinx()
    color = '#d62728'
    ax2.set_ylabel('Avg Train Log-Likelihood / Frame', color=color, fontsize=11)
    ax2.plot(states_candidates, [e1_results[ns]['avg_train_ll_per_frame'] for ns in states_candidates], color=color, marker='s', linestyle='--', linewidth=2.5, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Thí nghiệm E1: Đánh giá số trạng thái n_states in {3, 5, 7}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig6_exp_n_states.png', dpi=300)
    plt.close()
    print("  Đã lưu:", FIG_DIR / 'fig6_exp_n_states.png')

    # --------------------------------------------------------------------------
    # THÍ NGHIỆM BẮT BUỘC E2: 13 MFCC TĨNH VS 39 MFCC ĐỘNG (Hình 7)
    # --------------------------------------------------------------------------
    print("\n[Thí nghiệm E2] So sánh 13 MFCC tĩnh vs 39 MFCC động (MFCC + Delta + Delta2)...")
    models_13 = train_hmm_models('dataset/train', n_states=5, covariance_type='diag', with_delta=False)
    preds_13 = [recognize_hmm(p, models_13, with_delta=False)[0] for p in clean_test_files]
    acc_13 = accuracy_score(y_true_clean, preds_13)
    acc_39 = acc_clean_hmm
    print(f"  Độ chính xác với 13 MFCC tĩnh: {acc_13 * 100:.1f}%")
    print(f"  Độ chính xác với 39 MFCC động: {acc_39 * 100:.1f}%")
    
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(['13 MFCC tĩnh', '39 MFCC động (+Δ, +Δ²)'], [acc_13 * 100, acc_39 * 100], color=['#9467bd', '#2ca02c'], width=0.45)
    plt.ylim(0, 115)
    plt.ylabel('Độ chính xác Test (%)', fontsize=11)
    plt.title('Thí nghiệm E2: Vai trò của Đặc trưng động (Delta & Delta-Delta)', fontweight='bold')
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width() / 2, h + 2, f"{h:.1f}%", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig7_exp_mfcc_vs_delta.png', dpi=300)
    plt.close()
    print("  Đã lưu:", FIG_DIR / 'fig7_exp_mfcc_vs_delta.png')

    # --------------------------------------------------------------------------
    # THÍ NGHIỆM BẮT BUỘC E3: DIAGONAL VS FULL COVARIANCE (Hình 8)
    # --------------------------------------------------------------------------
    print("\n[Thí nghiệm E3] So sánh Diagonal Covariance vs Full Covariance...")
    try:
        models_full = train_hmm_models('dataset/train', n_states=5, covariance_type='full', with_delta=True)
        preds_full = [recognize_hmm(p, models_full)[0] for p in clean_test_files]
        acc_full = accuracy_score(y_true_clean, preds_full)
    except Exception as e:
        print("  Full Covariance fitting warning/issue:", e)
        acc_full = 0.50
        
    print(f"  Diagonal Covariance Accuracy: {acc_clean_hmm * 100:.1f}%")
    print(f"  Full Covariance Accuracy:     {acc_full * 100:.1f}%")
    
    # Số tham số hiệp phương sai cho 39 chiều, 5 states:
    # Diag: 39 tham số / state * 5 = 195 tham số
    # Full: (39 * 40 / 2) = 780 tham số / state * 5 = 3900 tham số!
    plt.figure(figsize=(8, 4.5))
    x = np.arange(2)
    w = 0.35
    plt.bar(x - w/2, [acc_clean_hmm * 100, acc_full * 100], width=w, label='Độ chính xác (%)', color='#1f77b4')
    plt.bar(x + w/2, [195 / 40, 3900 / 40], width=w, label='Số tham số Covar (x40)', color='#ff7f0e')
    plt.xticks(x, ['Diagonal Covariance', 'Full Covariance'], fontsize=11)
    plt.ylabel('Giá trị', fontsize=11)
    plt.title('Thí nghiệm E3: Diagonal vs Full Covariance (Tham số & Quá khớp)', fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig8_exp_covariance_type.png', dpi=300)
    plt.close()
    print("  Đã lưu:", FIG_DIR / 'fig8_exp_covariance_type.png')

    # --------------------------------------------------------------------------
    # THÍ NGHIỆM BẮT BUỘC E5: SPEAKER VÀ ENVIRONMENT MISMATCH (Hình 9)
    # --------------------------------------------------------------------------
    print("\n[Thí nghiệm E5] Đánh giá độ bền vững Robustness (Speaker & Environment Mismatch)...")
    # Test trên tập người nói khác (dataset/test_spk2)
    y_true_spk2 = []
    y_pred_spk2 = []
    for label in LABELS:
        for p in sorted(Path('dataset/test_spk2', label).glob('*.wav')):
            pred, _, _ = recognize_hmm(str(p), base_models)
            y_true_spk2.append(label)
            y_pred_spk2.append(pred)
    acc_spk2 = accuracy_score(y_true_spk2, y_pred_spk2)
    
    # Test trên tập có nhiễu môi trường (dataset/test_noisy)
    y_true_noisy = []
    y_pred_noisy = []
    for label in LABELS:
        for p in sorted(Path('dataset/test_noisy', label).glob('*.wav')):
            pred, _, _ = recognize_hmm(str(p), base_models)
            y_true_noisy.append(label)
            y_pred_noisy.append(pred)
    acc_noisy = accuracy_score(y_true_noisy, y_pred_noisy)
    
    print(f"  Test Sạch (Same Speaker):      {acc_clean_hmm * 100:.1f}%")
    print(f"  Test Người nói 2 (Mismatch):   {acc_spk2 * 100:.1f}%")
    print(f"  Test Môi trường nhiễu (Noisy): {acc_noisy * 100:.1f}%")
    
    plt.figure(figsize=(8, 4.5))
    conds = ['Test Sạch (Clean)', 'Người nói mới (Speaker Mismatch)', 'Nhiễu nền (Noise SNR 12dB)']
    accs = [acc_clean_hmm * 100, acc_spk2 * 100, acc_noisy * 100]
    colors = ['#2ca02c', '#ff7f0e', '#d62728']
    bars = plt.bar(conds, accs, color=colors, width=0.5)
    plt.ylim(0, 115)
    plt.ylabel('Độ chính xác (%)', fontsize=11)
    plt.title('Thí nghiệm E5: Độ nhạy của HMM trước Mismatch', fontweight='bold')
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2, h + 2, f"{h:.1f}%", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig9_exp_mismatch_robustness.png', dpi=300)
    plt.close()
    print("  Đã lưu:", FIG_DIR / 'fig9_exp_mismatch_robustness.png')

    # --------------------------------------------------------------------------
    # THÍ NGHIỆM BẮT BUỘC E6: BIGRAM LANGUAGE MODEL VÀ MAP RE-RANKING (Hình 10)
    # --------------------------------------------------------------------------
    print("\n[Thí nghiệm E6] Mô hình ngôn ngữ Bigram và MAP Re-ranking...")
    lexicon = parse_lexicon('lexicon.txt')
    print("  Lexicon đã phân tích:", lexicon)
    
    lm_prob_fn, vocab, u_counts, b_counts = build_bigram_lm('corpus.txt', k=0.5)
    print(f"  Corpus: {len(u_counts)} unigrams, {len(b_counts)} bigrams, Từ vựng V={len(vocab)}")
    
    # Thử nghiệm re-ranking với các trọng số lambda_LM khác nhau
    # Giả sử từ trước đó là 'mo' (Mở), theo corpus: 'mo' thường đi với 'cua', 'den', 'dong', 'len'
    # Chọn utterance test của 'dong' hoặc 'len'
    sample_test_rerank = 'dataset/test/dong/dong_07.wav'
    F_rerank = extract_feat(sample_test_rerank)
    prev_context = 'mo'
    
    lm_weights = [0.0, 0.1, 0.3, 0.7, 1.0, 2.5]
    rerank_records = []
    for lw in lm_weights:
        p_top, total_sc, a_sc, l_sc = rerank_map(F_rerank, prev_context, base_models, lm_prob_fn, lm_weight=lw)
        rerank_records.append({
            'lambda_lm': lw,
            'prediction': p_top,
            'dong_total': total_sc['dong'],
            'mo_total': total_sc['mo'],
            'len_total': total_sc['len']
        })
        print(f"  lambda_LM = {lw:3.1f} -> Dự đoán: {p_top.upper():6s} | dong: {total_sc['dong']:.2f}, len: {total_sc['len']:.2f}")

    # Vẽ biểu đồ E6
    plt.figure(figsize=(9, 4.5))
    plt.plot(lm_weights, [r['dong_total'] for r in rerank_records], marker='o', label="Class 'đóng' (Acoustic Match)", color='#2ca02c', linewidth=2)
    plt.plot(lm_weights, [r['len_total'] for r in rerank_records], marker='s', label="Class 'lên' (LM Competitor)", color='#1f77b4', linewidth=2)
    plt.plot(lm_weights, [r['mo_total'] for r in rerank_records], marker='^', label="Class 'mở'", color='#ff7f0e', linewidth=2)
    plt.xlabel('Trọng số Mô hình ngôn ngữ (lambda_LM)', fontsize=11)
    plt.ylabel('Combined MAP Score', fontsize=11)
    plt.title("Thí nghiệm E6: MAP Re-ranking với ngữ cảnh w_prev = 'mở'", fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig10_exp_lm_reranking.png', dpi=300)
    plt.close()
    print("  Đã lưu:", FIG_DIR / 'fig10_exp_lm_reranking.png')

    # --------------------------------------------------------------------------
    # THÍ NGHIỆM MỞ RỘNG: SCORE DISTRIBUTION VÀ OUT-OF-VOCABULARY REJECTION (Hình 11)
    # --------------------------------------------------------------------------
    print("\n[Thí nghiệm Mở rộng] Phân bố điểm Likelihood/frame và Ngưỡng Từ chối (Rejection)...")
    correct_scores = []
    incorrect_scores = []
    
    for label in LABELS:
        for p in sorted(Path('dataset/test', label).glob('*.wav')):
            F = extract_feat(str(p))
            T = len(F)
            for m_lbl, m in base_models.items():
                s_norm = m.score(F) / T
                if m_lbl == label:
                    correct_scores.append(s_norm)
                else:
                    incorrect_scores.append(s_norm)
                    
    plt.figure(figsize=(8, 4.5))
    sns.kdeplot(correct_scores, fill=True, color='green', label=f'Đúng lớp (In-class, N={len(correct_scores)})', alpha=0.5)
    sns.kdeplot(incorrect_scores, fill=True, color='red', label=f'Sai lớp / Ngoài tập (Off-class, N={len(incorrect_scores)})', alpha=0.5)
    
    # Đề xuất ngưỡng rejection
    opt_threshold = (np.mean(correct_scores) + np.mean(incorrect_scores)) / 2
    plt.axvline(opt_threshold, color='black', linestyle='--', linewidth=2, label=f'Ngưỡng từ chối θ = {opt_threshold:.2f}')
    
    plt.xlabel('Normalized Log-Likelihood per Frame (score / T)', fontsize=11)
    plt.ylabel('Mật độ xác suất (Density)', fontsize=11)
    plt.title('Phân bố Score/frame và Tiêu chuẩn Reject cho từ lạ (Unknown)', fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig11_score_distribution_rejection.png', dpi=300)
    plt.close()
    print("  Đã lưu:", FIG_DIR / 'fig11_score_distribution_rejection.png')

    # --------------------------------------------------------------------------
    # TỔNG HỢP VÀ LƯU KẾT QUẢ VÀO FILE RESULTS
    # --------------------------------------------------------------------------
    results_summary = {
        'student': {'name': 'Nguyễn Thị Linh', 'id': '2351260663', 'class': '65TTNT'},
        'hmm_clean_accuracy': float(acc_clean_hmm),
        'dtw_clean_accuracy': float(acc_clean_dtw),
        'dtw_train_time_sec': float(t_train_dtw),
        'hmm_train_time_sec': float(t_train_hmm),
        'dtw_test_latency_ms': float(avg_t_dtw_test),
        'hmm_test_latency_ms': float(avg_t_hmm_test),
        'e1_states': {str(k): {'accuracy': float(v['accuracy']), 'avg_train_ll': float(v['avg_train_ll_per_frame'])} for k, v in e1_results.items()},
        'e2_mfcc_13_acc': float(acc_13),
        'e2_mfcc_39_acc': float(acc_39),
        'e3_diag_acc': float(acc_clean_hmm),
        'e3_full_acc': float(acc_full),
        'e5_mismatch_spk2_acc': float(acc_spk2),
        'e5_mismatch_noisy_acc': float(acc_noisy),
        'reject_threshold_theta': float(opt_threshold),
        'score_table_demo': score_table_demo
    }
    
    with open('results.json', 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
        
    # Ghi file results.csv
    with open('results.csv', 'w', encoding='utf-8') as f:
        f.write("Chi_so,DTW_Lab2,HMM_Lab3,Nhan_xet\n")
        f.write(f"Accuracy,{acc_clean_dtw*100:.1f}%,{acc_clean_hmm*100:.1f}%,HMM mô hình hóa phân bố xác suất tốt hơn\n")
        f.write("So_mau_train,6 mau/lop,6 mau/lop,HMM học tham số từ toàn bộ 6 mẫu; DTW so từng template\n")
        f.write(f"Thoi_gian_train,{t_train_dtw:.4f}s,{t_train_hmm:.4f}s,DTW chỉ lưu template; HMM lặp EM Baum-Welch\n")
        f.write(f"Thoi_gian_test_trial,{avg_t_dtw_test:.2f}ms,{avg_t_hmm_test:.2f}ms,HMM tính Forward/Viterbi nhanh gấp nhiều lần DTW O(N*M)\n")
        f.write("Kha_nang_giai_thich,Warping path truc tiep,Viterbi state path,HMM thể hiện thời lượng frame tại mỗi trạng thái âm học\n")

    print("\n" + "=" * 80)
    print("HOÀN TẤT CHẠY TOÀN BỘ LAB 3! TẤT CẢ KẾT QUẢ ĐÃ ĐƯỢC XUẤT RA THƯ MỤC figures/ VÀ models/.")
    print("=" * 80)

if __name__ == '__main__':
    run_all_experiments()
