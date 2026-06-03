# 🧮 Vietnamese GPT-2 Math Fine-tuning

Fine-tuning mô hình **GPT-2 Vietnamese** (124M params) để giải bài toán có lời văn bằng tiếng Việt, sử dụng pipeline hybrid 7 tầng: **Answer-Only LoRA + Checkpoint Ensemble + Numeric Constrained Decoding + Hybrid Retrieval**.

> **Kết quả:** 5.617 / 10 trên validation 1,000 mẫu — tăng gấp ~8× so với baseline full-solution SFT.

---

## 📋 Mục lục

- [Bài toán](#-bài-toán)
- [Kiến trúc Pipeline](#-kiến-trúc-pipeline)
- [Cấu trúc Repository](#-cấu-trúc-repository)
- [Cài đặt và Chạy](#-cài-đặt-và-chạy)
- [Dataset](#-dataset)
- [Kết quả](#-kết-quả)
- [Kỹ thuật chính](#-kỹ-thuật-chính)

---

## 🎯 Bài toán

**Input:** Bài toán lời văn tiếng Việt  
**Output:** Một con số duy nhất làm đáp án

**Ví dụ:**
```
Câu hỏi: Bridgette mời 84 khách, Alex mời 2/3 số đó.
         Mỗi khách ăn 8 ngọn măng tây. Người phục vụ làm thêm 10 đĩa.
         Hỏi cần bao nhiêu ngọn măng tây?

Đáp án:  1200
```

**Cách chấm điểm** — dựa trên sai số tương đối (relative error):

| Sai số | Điểm | Ý nghĩa |
|---|---:|---|
| ≤ 1% | 10 | Chính xác |
| ≤ 10% | 5 | Gần đúng |
| ≤ 50% | 1 | Sai nhiều |
| > 50% | 0 | Sai hoàn toàn |

**Thách thức:** GPT-2 là mô hình **ngôn ngữ**, không phải máy tính. Nó dự đoán token tiếp theo dựa trên pattern, không biết thực sự tính toán. Pipeline phải thiết kế để tận dụng tối đa khả năng pattern matching này.

---

## 🏗️ Kiến trúc Pipeline

```
┌─────────────────────────────────────────────────────────┐
│ 1. DATA CLEANING & SMART SPLIT                          │
│    100K mẫu → clean/dedup → 56,651 → train 45,827      │
│                                        valid 1,000      │
├─────────────────────────────────────────────────────────┤
│ 2. ANSWER-ONLY TARGET                                   │
│    Model CHỈ học sinh số "1200" thay vì lời giải dài   │
│    Prompt: "Dạng: GSM\nBài toán: ...\nĐáp án là:"     │
├─────────────────────────────────────────────────────────┤
│ 3. LoRA FINE-TUNING                                     │
│    Chỉ train 3.65% tham số (4.7M / 129M)              │
│    LR=1e-3, 13 epochs trên 2×T4 GPU                   │
├─────────────────────────────────────────────────────────┤
│ 4. CHECKPOINT SELECTION                                 │
│    Đánh giá 13 checkpoints → Best = epoch 8 (5.77/10) │
├─────────────────────────────────────────────────────────┤
│ 5. CHECKPOINT ENSEMBLE                                  │
│    Top 3 checkpoint vote majority → giảm variance      │
├─────────────────────────────────────────────────────────┤
│ 6. NUMERIC CONSTRAINED DECODING                         │
│    Ép model chỉ sinh token số → 100% extractable       │
├─────────────────────────────────────────────────────────┤
│ 7. HYBRID RETRIEVAL                                     │
│    Tìm bài tương tự trong train → majority vote        │
│    Sửa 211/1000 câu → +0.486 điểm                     │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Cấu trúc Repository

```
Vietnamese-gpt2-math-finetuning/
├── fine-tune-gpt-2-for-math.ipynb   # 📓 Notebook chính (toàn bộ pipeline)
├── README.md                         # 📄 File này
├── .gitignore                        # Git ignore
├── NlpHUSTgpt2-vietnamese/           # 🤖 Model pretrained (download riêng)
│   ├── config.json
│   ├── tokenizer.json
│   ├── vocab.json
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── added_tokens.json
│   ├── tokenizer_config.json
│   └── pytorch_model.bin             # ⚠️ ~510MB, không push lên GitHub
└── dataset_math/                     # 📊 Dataset (download riêng)
    ├── train.json                    # ⚠️ ~218MB, 100K mẫu
    └── valid.json                    # ~2.3MB, 1K mẫu
```

---

## 🚀 Cài đặt và Chạy

### Yêu cầu hệ thống

- **GPU:** 1-2× NVIDIA T4 (hoặc tương đương, ≥16GB VRAM)
- **RAM:** ≥16GB
- **Python:** ≥ 3.10
- **Thời gian chạy:** ~2h42m (trên 2×T4)

### Bước 1: Clone repo

```bash
git clone https://github.com/MILKPLZ/Vietnamese-gpt2-math-finetuning.git
cd Vietnamese-gpt2-math-finetuning
```

### Bước 2: Cài dependencies

```bash
pip install torch transformers peft datasets tqdm
```

### Bước 3: Download model & dataset

**Model GPT-2 Vietnamese:**
```bash
# Từ Hugging Face
git lfs install
git clone https://huggingface.co/NlpHUST/gpt2-vietnamese NlpHUSTgpt2-vietnamese/
```

**Dataset:** Tải `train.json` và `valid.json` vào thư mục `dataset_math/` (từ Kaggle dataset hoặc nguồn cung cấp).

### Bước 4: Chạy notebook

Mở `fine-tune-gpt-2-for-math.ipynb` trên **Kaggle** (khuyến nghị, có sẵn GPU) hoặc Jupyter local, rồi chạy tuần tự tất cả các cell.

**Trên Kaggle:**
1. Upload notebook lên Kaggle
2. Attach dataset `dataset_math` và model `NlpHUSTgpt2-vietnamese` làm Input
3. Chọn GPU T4 ×2, Internet OFF
4. Run All

---

## 📊 Dataset

| | Train | Valid |
|---|---:|---:|
| Số mẫu | 100,000 | 1,000 |
| Mẫu sạch (sau dedup) | 56,651 | — |
| Train split | 45,827 | — |
| Heldout valid | 1,000 (balanced) | — |

**8 loại bài toán:**

| Type | Mô tả | Độ khó |
|---|---|---|
| GSM_AnsAug | Bài GSM gốc | ⭐⭐ |
| GSM_Rephrased | Bài GSM viết lại | ⭐ |
| GSM_FOBAR | Bài GSM tìm biến x | ⭐⭐⭐⭐ |
| GSM_SV | Bài GSM tìm giá trị | ⭐⭐⭐⭐ |
| MATH_AnsAug | Bài MATH gốc | ⭐⭐⭐ |
| MATH_Rephrased | Bài MATH viết lại | ⭐⭐ |
| MATH_FOBAR | Bài MATH tìm biến x | ⭐⭐⭐⭐⭐ |
| MATH_SV | Bài MATH tìm giá trị | ⭐⭐⭐⭐⭐ |

---

## 📈 Kết quả

### Tổng quan

| Cấu hình | Score /10 |
|---|---:|
| Baseline: Full-solution SFT | 0.585 |
| Model-only (LoRA + ensemble) | 5.131 |
| **+ Hybrid retrieval (final)** | **5.617** |

### Theo loại bài

| Type | Score /10 | Exact (10 điểm) % |
|---|---:|---:|
| GSM_Rephrased | 8.79 | 86.4% |
| MATH_Rephrased | 8.00 | 78.4% |
| MATH_AnsAug | 5.93 | 56.8% |
| GSM_AnsAug | 5.50 | 51.2% |
| MATH_SV | 4.88 | 45.6% |
| GSM_SV | 4.61 | 44.8% |
| MATH_FOBAR | 3.85 | 36.0% |
| GSM_FOBAR | 3.38 | 31.2% |

### Training metrics

| Metric | Giá trị |
|---|---|
| Trainable params | 4,718,592 (3.65%) |
| Training loss | 10.5 → 0.02 |
| Best checkpoint | Epoch 8 (5.77/10) |
| Extractability | 100% |
| Runtime | ~2h42m (2×T4) |

---

## 🔧 Kỹ thuật chính

### 1. Answer-Only Target
Thay vì dạy model sinh lời giải dài (400+ token), chỉ dạy sinh đáp án (`" 1200"`, 2-5 token). Giải quyết answer-signal dilution — focus 100% loss vào phần metric cần.

### 2. LoRA (Low-Rank Adaptation)
Chỉ train 3.65% tham số bằng adapter rank-32 trên `c_attn`, `c_proj`, `c_fc`. Cho phép LR cao (1e-3) và train nhiều epoch (13) mà không overfit nặng.

### 3. Checkpoint Selection
Đánh giá **tất cả 13 checkpoints** bằng inference thực trên 300 mẫu heldout. Best = epoch 8, KHÔNG phải epoch cuối (13). Tránh dùng checkpoint overfit.

### 4. Checkpoint Ensemble
Top-3 checkpoints sinh đáp án song song → **majority vote** chọn đáp án được ≥2/3 đồng ý. Giảm variance, tăng ổn định.

### 5. Numeric Constrained Decoding
Ép model chỉ sinh token thuộc `{0-9, -, +, ., ,, /, %}`. Kết hợp `StopAfterAnswerNumber` dừng sớm. Đạt 100% extractability.

### 6. Hybrid Retrieval
- Build index từ train theo `source_group_key`
- Tìm bài cùng source → **majority vote** đáp án
- **Logprob arbitration:** chỉ dùng retrieval khi model kém tự tin
- Sửa 211/1000 câu, +0.486 điểm

---

## 📚 Cấu trúc Notebook

| Cell | Nội dung | Thời gian |
|---|---|---|
| 0 | Import & environment check | ~30s |
| 1 | Config (tất cả hyperparams) | ~1s |
| 2 | Tokenizer smoke-check | ~5s |
| 3 | Load data | ~10s |
| 4 | Data cleaning & split | ~10s |
| 5 | SFTDataset & PadCollator | ~1s |
| 6 | Evaluation utilities | ~1s |
| 7 | Inference engine (decoding, retrieval) | ~1s |
| 8 | **Training (LoRA, 13 epochs)** | **~2h23m** |
| 9 | Inference & evaluation | ~2.5m |
| 10 | Error analysis | ~1s |
| 11 | Phase 2 (test predictions) | skipped (phase1) |
| 12 | List saved artifacts | ~1s |

---

## 📖 Tài liệu tham khảo

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
- [GSM8K: Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168) (Cobbe et al., 2021)
- [MATH Dataset](https://arxiv.org/abs/2103.03874) (Hendrycks et al., 2021)
- [NlpHUST/gpt2-vietnamese](https://huggingface.co/NlpHUST/gpt2-vietnamese)
- [Hugging Face PEFT](https://huggingface.co/docs/peft)

---

## 📝 License

Dự án phục vụ mục đích học tập (BTL Deep Learning).
