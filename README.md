# 🧮 Vietnamese GPT-2 Math Fine-tuning

Fine-tuning mô hình **GPT-2 Vietnamese** (124M params) để giải bài toán có lời văn bằng tiếng Việt, sử dụng pipeline hybrid: **Answer-Only LoRA + Checkpoint Selection + Hybrid Retrieval**.

> **Kết quả:** 5.643 / 10 trên tập official validation 1,000 mẫu (retrieval: 5.886).

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
│ 1. DATA CLEANING & DEDUP                                │
│    95,400 mẫu → clean/dedup → 56,651 mẫu sạch           │
├─────────────────────────────────────────────────────────┤
│ 2. ANSWER-ONLY TARGET                                   │
│    Model CHỈ học sinh số "1200" thay vì lời giải dài    │
│    Prompt: "Dạng: GSM\nBài toán: ...\nĐáp án là:"       │
├─────────────────────────────────────────────────────────┤
│ 3. LoRA FINE-TUNING                                     │
│    Chỉ train 3.65% tham số (4.7M / 129M)                │
│    r=32, α=64, LR=1e-3, 12 epochs trên 2×T4 GPU         │
├─────────────────────────────────────────────────────────┤
│ 4. CHECKPOINT SELECTION (trên official valid)           │
│    Đánh giá 12 checkpoints × 1000 mẫu valid             │
│    Best = epoch 11, checkpoint-9746 (5.643/10)          │
├─────────────────────────────────────────────────────────┤
│ 5. NUMERIC CONSTRAINED DECODING                         │
│    Ép model chỉ sinh token số → 100% extractable        │
├─────────────────────────────────────────────────────────┤
│ 6. SAFE HYBRID RETRIEVAL                                │
│    Source-group matching + type-aware gating            │
│    Retrieval sửa 301/1000 câu → +0.243 điểm             │
│    Sửa 301/1000 câu → +0.243 điểm                       │
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
    ├── train.json                    # ⚠️ ~218MB, 95,400 mẫu
    └── valid.json                    # ~2.3MB, 1,000 mẫu
```

---

## 🚀 Cài đặt và Chạy

### Yêu cầu hệ thống

- **GPU:** 2× NVIDIA T4 (hoặc tương đương, ≥16GB VRAM)
- **RAM:** ≥16GB
- **Python:** ≥ 3.10
- **Thời gian chạy:** ~2h52m (training ~2h42m + checkpoint eval ~10m + inference ~1m)

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
git lfs install
git clone https://huggingface.co/NlpHUST/gpt2-vietnamese NlpHUSTgpt2-vietnamese/
```

**Dataset:** Tải `train.json` và `valid.json` vào thư mục `dataset_math/` (từ Kaggle dataset).

### Bước 4: Chạy notebook

Mở `fine-tune-gpt-2-for-math.ipynb` trên **Kaggle** (khuyến nghị) hoặc Jupyter local.

**Trên Kaggle:**
1. Upload notebook lên Kaggle
2. Attach dataset `dataset_math` và model `NlpHUSTgpt2-vietnamese` làm Input
3. Chọn GPU T4 ×2, Internet OFF
4. Run All

---

## 📊 Dataset

| | Giá trị |
|---|---:|
| Train (raw) | 95,400 mẫu |
| Train (sau clean + dedup) | 56,651 mẫu |
| Valid (official) | 1,000 mẫu |

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

| Cấu hình | Score /10 | Exact(10) | Near(5) | Partial(1) | Fail(0) |
|---|---:|---:|---:|---:|---:|
| Model-only (best checkpoint) | 5.643 | 539 | 21 | 148 | 292 |
| **+ Safe Hybrid Retrieval (final)** | **5.886** | **565** | **19** | **141** | **275** |

### Theo loại bài (final hybrid)

| Type | N | Score /10 | Exact(10) | Extractable % |
|---|---:|---:|---:|---:|
| GSM_Rephrased | 197 | **9.36** | 182 | 100% |
| MATH_Rephrased | 116 | **7.86** | 91 | 100% |
| GSM_AnsAug | 209 | 5.39 | 106 | 100% |
| GSM_SV | 97 | 4.93 | 45 | 100% |
| MATH_AnsAug | 173 | 4.42 | 72 | 99.4% |
| MATH_SV | 41 | 3.98 | 15 | 100% |
| MATH_FOBAR | 45 | 3.93 | 16 | 100% |
| GSM_FOBAR | 122 | 3.46 | 38 | 100% |

### Checkpoint Selection (model-only, trên 1000 mẫu official valid)

| Epoch | Checkpoint | Score /10 | Exact(10) |
|---|---|---:|---:|
| 1 | checkpoint-886 | 1.114 | 74 |
| 2 | checkpoint-1772 | 1.503 | 106 |
| 3 | checkpoint-2658 | — | — |
| 4 | checkpoint-3544 | — | — |
| 5 | checkpoint-4430 | — | — |
| 6 | checkpoint-5316 | 4.652 | 436 |
| 7 | checkpoint-6202 | 4.920 | 458 |
| 8 | checkpoint-7088 | 5.253 | 497 |
| 9 | checkpoint-7974 | 5.426 | 515 |
| 10 | checkpoint-8860 | 5.573 | 532 |
| **11** | **checkpoint-9746** | **5.643** ⭐ | **539** |
| 12 | checkpoint-10632 | 5.631 | 537 |

### Training metrics

| Metric | Giá trị |
|---|---|
| Trainable params | 4,718,592 (3.65%) |
| Training loss | 10.5 → 0.024 |
| Best checkpoint | Epoch 11 — checkpoint-9746 |
| Steps/epoch | 886 |
| Total training steps | 10,632 |
| Training wall time | 161.72 min (~2h42m) |
| GPU | 2× Tesla T4 |
| Batch size | 16/GPU × 2 accum = 64 effective |

---

## 🔧 Kỹ thuật chính

### 1. Answer-Only Target
Thay vì dạy model sinh lời giải dài (400+ ký tự), chỉ dạy sinh đáp án (`" 1200"`, 2–5 token). Giải quyết answer-signal dilution — focus 100% loss vào phần metric cần.

### 2. LoRA (Low-Rank Adaptation)
Chỉ train 3.65% tham số bằng adapter rank-32 trên `c_attn`, `c_proj`, `c_fc`. Cho phép LR cao (1e-3) và train nhiều epoch (12) mà không overfit nặng. Merge LoRA adapter vào model gốc sau training.

### 3. Checkpoint Selection trên Official Valid
Đánh giá **tất cả 12 checkpoints** bằng inference thực trên **1000 mẫu official valid** (không dùng heldout riêng). Best = epoch 11 (5.643/10), không phải epoch 12 (5.631/10).

### 4. Numeric Constrained Decoding
Ép model chỉ sinh token thuộc `{0-9, -, +, ., ,, /, %}`. Kết hợp stopping criteria dừng sớm khi phát hiện số. Đạt gần 100% extractability (999/1000).

### 5. Safe Hybrid Retrieval
Pipeline retrieval **bảo thủ** — chỉ can thiệp khi chắc chắn:
- Build index từ train theo `source_group_key` (trường `original_question_en`)
- **Retrieval-first** cho type ổn định: Rephrased, AnsAug — ghi đè model nếu majority ≥ 50%
- **Model-first** cho type khó: FOBAR, SV — giữ model, chỉ override khi model agrees
- Fallback tự động: Dễ dàng sử dụng model-only hoặc dùng retrieval.

**Kết quả:** Hybrid tăng cho model-only từ 5.643 → 5.886 (+0.243). Nhưng mà do không được can thiệp vào các nguồn original của câu hỏi nên lúc chạy sẽ set up model only.

---

## 📚 Cấu trúc Notebook

| Cell | Nội dung | Thời gian |
|---|---|---|
| 0 | Markdown giới thiệu | — |
| 1 | Import & environment check | ~30s |
| 2 | Config (tất cả hyperparams) | ~1s |
| 3 | Tokenizer smoke-check | ~1s |
| 4 | Load data | ~3s |
| 5 | Data cleaning, dedup & target build | ~5s |
| 6 | SFTDataset & PadCollator | ~1s |
| 7 | Evaluation utilities | ~1s |
| 8 | Inference engine (decoding, retrieval) | ~1s |
| 9 | Training functions | ~1s |
| 10 | **Training (LoRA, 12 epochs) + Checkpoint Selection** | **~2h42m + ~10m** |
| 11 | Final inference & hybrid retrieval | ~2m |
| 12 | Error analysis | ~1s |
| 13 | Phase 2 (test predictions) | skipped |
| 14 | List saved artifacts | ~1s |

---

## 📖 Tài liệu tham khảo

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
- [GSM8K: Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168) (Cobbe et al., 2021)
- [MATH Dataset](https://arxiv.org/abs/2103.03874) (Hendrycks et al., 2021)
- [NlpHUST/gpt2-vietnamese](https://huggingface.co/NlpHUST/gpt2-vietnamese)
- [Hugging Face PEFT](https://huggingface.co/docs/peft)
