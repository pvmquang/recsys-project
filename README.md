# Hệ gợi ý phim lai ghép — đồ án IT4613

Nghiên cứu so sánh các kỹ thuật gợi ý trên MovieLens + TMDB, từ baseline
đến mô hình lai ghép, đo trên cùng một giao thức đánh giá.

## Chạy thử

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements-lock.txt   # đúng phiên bản đã ghim, ba máy như nhau
pip install -e . --no-deps             # cài gói recsys, script import được từ mọi nơi

pytest -q                       # 264 test, không gọi mạng
python scripts/run_eval.py      # tự tải MovieLens ở lần chạy đầu
```

Cài bằng `requirements-lock.txt`, **không** bằng `requirements.txt`. File thứ hai
chỉ ghim gói trực tiếp, còn gói kéo theo (`threadpoolctl`, `joblib`…) vẫn có thể
lệch giữa ba máy. Thêm thư viện mới thì sửa `requirements.txt`, cài, rồi ghim lại:
`pip freeze --exclude recsys > requirements-lock.txt`.

GitHub Actions chạy `pytest` trên Ubuntu và Windows mỗi lần push lên `main` hoặc mở PR
([.github/workflows/test.yml](.github/workflows/test.yml)).

## Kết quả hiện có (ml-latest-small, seed 42)

```bash
python scripts/so_lieu_slide.py   # mọi số dưới đây, kèm CI 95%
```

| Mô hình | recall@10 | ndcg@10 | MRR | AUC | coverage@10 |
|---|---|---|---|---|---|
| B0-Random | 0.1180 | 0.0538 | 0.0561 | 0.4956 | 0.4664 |
| B0-Popularity | 0.5967 | 0.3590 | 0.3019 | 0.8315 | 0.0126 |
| B1-ItemKNN (k=80) | 0.6803 | 0.4761 | 0.4225 | 0.8342 | 0.0749 |
| B2-UserKNN (k=40) | 0.6443 | 0.4214 | 0.3647 | 0.8310 | 0.0288 |
| B3-MF implicit (f=32) | 0.7279 | 0.5003 | 0.4386 | 0.8605 | 0.1006 |
| B4-BPR (f=64) | 0.7328 | 0.4886 | 0.4239 | 0.8858 | 0.0911 |
| B5-Content (TMDB) | 0.4869 | 0.2563 | 0.2067 | 0.8052 | 0.2788 |
| B6-Rules (luật theo phiên) | 0.7213 | 0.4780 | 0.4131 | 0.8689 | 0.1069 |
| B7-Hybrid (BPR + B5, α theo lịch sử) | **0.7656** | **0.5215** | **0.4561** | **0.9170** | 0.0983 |
| B8-LightGCN (K=4) | 0.7574 | 0.5112 | 0.4436 | 0.8858 | 0.1055 |

Số in đậm là cao nhất cột (trừ coverage). B3 và B4 hòa về recall@10 và nDCG@10 (CI ghép
cặp chứa 0). B4 hơn B3 về AUC có ý nghĩa. B7 hơn B3 và B4 về nDCG@10 có ý nghĩa ở bảng
này, nhưng trên toàn danh mục thì là nhiễu. Coverage của B7 chia cho 9.742 phim, vì B7
gợi ý được cả phim chưa có trong train. B8 hơn B4 về nDCG@10 ở bảng này, sát ngưỡng, và
hòa B3. Chưa đo ghép cặp B8 với B7.

Số láng giềng k và siêu tham số của MF, BPR, B5, B6, B7, B8 chọn trên tập validation (xem
[Giao thức](#giao-thức-đánh-giá)). Dòng B3-B8 lấy từ `scripts/exp_mf.py`,
`scripts/exp_bpr.py`, `scripts/exp_content.py`, `scripts/exp_rules.py`,
`scripts/exp_hybrid.py` và `scripts/exp_lightgcn.py`, xem [phần B3](#bậc-b3-phân-rã-ma-trận),
[phần B4](#bậc-b4-bpr-mf), [phần B5](#bậc-b5-gợi-ý-theo-nội-dung),
[phần B6](#bậc-b6-luật-kết-hợp-theo-phiên), [phần B7](#bậc-b7-lai-ghép) và
[phần B8](#bậc-b8-lightgcn).
AUC tính từ điểm, cặp bằng điểm tính 0.5. Coverage là coverage@10, cùng k với
các cột bên cạnh.

**Cả bảng này đo trên 1 phim đúng + 99 phim âm lấy mẫu.** Trên toàn danh mục,
chỉ còn ItemKNN, MF và BPR hơn Popularity có ý nghĩa, và đều sát ngưỡng. Ba mô hình
này hòa nhau. B7 cũng hơn Popularity (+0.019 nDCG@10, CI [+0.007, +0.031]) và hòa MF,
BPR. B8 cũng vậy: hơn Popularity +0.019, CI [+0.008, +0.030], hòa MF, BPR, ItemKNN. Xem
[phần đo thêm ở dưới](#đo-thêm-toàn-danh-mục-auc-cold-start).

**Kiểm tra bằng Random không phát hiện lỗi khung đo.** Qua 10 seed: recall@10
= 0.101 ± 0.012, AUC = 0.503 ± 0.012, khớp kỳ vọng lý thuyết 0.10 và 0.50.
Nếu sau này sửa gì mà hai số này lệch đi, lỗi nằm ở khung đo chứ không ở mô hình.

**Với giao thức lấy mẫu, ItemKNN hơn Popularity cả về độ chính xác lẫn độ phủ.**
nDCG@10 hơn +0.117 (+33%, CI 95% [+0.097, +0.139]), phủ danh mục gấp 6 lần
(7,5% so với 1,26%). ItemKNN cũng hơn UserKNN: +0.055 nDCG@10, CI [+0.034, +0.074],
và +0.036 recall@10, CI [+0.007, +0.067]. Độ phủ không phụ thuộc giao thức, còn
nDCG thì có.

**AUC: ba cặp đều là nhiễu.** Bootstrap ghép cặp theo user:

| so sánh AUC | chênh lệch | CI 95% | |
|---|---|---|---|
| ItemKNN − Popularity | +0.003 | [−0.013, +0.018] | nhiễu |
| UserKNN − Popularity | −0.001 | [−0.014, +0.014] | nhiễu |
| ItemKNN − UserKNN | +0.003 | [−0.013, +0.019] | nhiễu |

Với k = 40, cặp cuối từng có ý nghĩa (−0.018). Đổi sang k = 80 thì mất, vì ItemKNN
chấm được nhiều phim đúng hơn: tỉ lệ phim đúng bị `-inf` giảm từ 29% xuống 23%.
AUC ở giao thức này chủ yếu đo *"có ý kiến về phim đúng không"*. Đừng dùng nó để
xếp hạng mô hình.

## Thêm một bậc mô hình

Toàn bộ chín bậc B0–B8 cài đặt chung một hợp đồng ở [`recsys/base.py`](recsys/base.py):

```python
class MyModel(Recommender):
    name = "B3-MF"

    def _fit(self, train): ...                    # train: user_id, item_id, rating, timestamp
    def score(self, user_id, item_ids): ...       # trả mảng điểm, cao = gợi ý trước
    def explain(self, user_id, item_id): ...      # câu giải thích (chuẩn đầu ra M2)
```

`recommend()` được suy ra từ `score()`, không cần viết lại. Viết xong thì
thêm một dòng vào `MODELS` trong `scripts/run_eval.py` là đo được ngay.

**Đừng đổi hợp đồng này sau tuần 2.** Mỗi lần đổi là phải sửa lại mọi bậc
đã viết xong, và ba nhánh đang làm song song sẽ không khớp nhau.

Mô hình có siêu tham số thì chọn trên validation, theo cách `exp_knn.py` (thí
nghiệm 3) chọn k:

```python
val = split.validation()                              # tương tác áp chót
for k in (...):
    evaluate(Model(k).fit(val.train), val)            # chọn ở đây
evaluate(Model(k_tot).fit(split.train), split)        # báo cáo ở đây
```

## Giao thức đánh giá

Leave-one-out **theo thời gian**: tương tác cuối cùng của mỗi user vào tập
kiểm tra, chấm điểm trên 1 phim đúng trộn với 99 phim lấy ngẫu nhiên.

Không dùng chia ngẫu nhiên, vì nó cho mô hình nhìn thấy tương lai của chính
user đó, làm mọi chỉ số cao lên một cách giả tạo.

Ba quyết định cả nhóm đã chốt (checklist 2.1-2.3):

- **Validation là tương tác áp chót** của mỗi user. Siêu tham số chọn trên
  validation, rồi huấn luyện trên `split.train` (bằng train + validation) và đo
  trên test. Không bao giờ chọn trên test.
- **Phá hòa timestamp ngẫu nhiên theo seed.** 94/610 user có nhiều phim trùng
  timestamp cuối (nhiều nhất 34 phim cùng một giây). Trước đây phim test luôn là
  phim có `item_id` lớn nhất, tức lệch về phim mới thêm vào MovieLens. Đổi cách
  này làm đổi phim test của 59 user.
- **Bài toán là dự đoán tương tác, không phải dự đoán sở thích.** 16,2% phim
  test bị user chấm dưới 3 sao (12,6% từ 2 sao trở xuống). Nhóm giữ nguyên dữ
  liệu, khớp với các bài báo top-K trên MovieLens. Báo cáo không được viết "đoán
  đúng phim user thích".

Hạn chế còn lại, phải nêu trong báo cáo:
- Train vẫn chứa tương tác về sau của user **khác** (ví dụ số đếm của
  Popularity). Giao thức chỉ chặn tương lai *của chính user đó*.
- 23/610 phim test chưa từng xuất hiện trong train, nên không mô hình lọc cộng
  tác nào đoán được.

`seed` mặc định 42 và **bắt buộc giữ nguyên giữa ba thành viên**. Ba người
chạy trên ba lần chia khác nhau thì số của ba người không so sánh được, và
không có gì trên bảng kết quả nói ra điều đó.

> **Recall@K = HitRate@K và MAP@K = MRR** ở giao thức này, vì mỗi user chỉ
> giữ lại một item đúng. Bày chúng như ba bằng chứng độc lập trong báo cáo
> là nói quá. Chi tiết ở đầu [`recsys/eval/metrics.py`](recsys/eval/metrics.py).

## Ba thí nghiệm của bậc B1/B2

```bash
python scripts/exp_knn.py
```

Mỗi thí nghiệm ghi một file CSV trong `reports/`.

### 1. Theo nhóm user: ItemKNN thắng ở cả ba nhóm

nDCG@10. Nhóm chia theo **phân vị** số tương tác trong train, mỗi nhóm ~200 user:

| nhóm (số tương tác) | số user | Popularity | ItemKNN | UserKNN | ItemKNN − UserKNN, CI 95% |
|---|---|---|---|---|---|
| 19-44 | 209 | 0.4181 | **0.5612** | 0.5043 | +0.057 [+0.016, +0.096] |
| 45-126 | 198 | 0.3989 | **0.5321** | 0.4695 | +0.063 [+0.028, +0.098] |
| 127-2697 | 203 | 0.2591 | **0.3337** | 0.2892 | +0.045 [+0.015, +0.076] |

- ItemKNN thắng UserKNN ở cả ba nhóm, chênh lệch có ý nghĩa. Popularity thấp
  nhất ở mọi nhóm.
- Mọi mô hình đều kém nhất ở nhóm nhiều lịch sử nhất. Độ khó của user nhiều
  lịch sử là tính chất của dữ liệu, không riêng gì Popularity.
- **Đây không phải phân tích cold-start**: nhóm thưa nhất vẫn có ≥ 19 tương tác.
  Cold-start thật nằm ở [phần đo thêm](#đo-thêm-toàn-danh-mục-auc-cold-start).

### 2. Hai chế độ chấm điểm

| | recall@10 | ndcg@10 | MRR | coverage@10 |
|---|---|---|---|---|
| ItemKNN `sum` | 0.6803 | **0.4761** | 0.4225 | 0.0749 |
| ItemKNN `weighted` | 0.5787 | 0.2651 | 0.1872 | **0.2859** |
| UserKNN `sum` | 0.6443 | **0.4214** | 0.3647 | 0.0288 |
| UserKNN `weighted` | 0.4525 | 0.2089 | 0.1604 | **0.1138** |

`weighted` là công thức **dự đoán rating** ở slide L2 tr.17/22. Nó tụt 44-50%
nDCG nhưng phủ danh mục gấp 3,8-4 lần. Lý do: phép chia cho `Σ sim` xoá mất tín
hiệu *"có bao nhiêu bằng chứng"*, nên một phim ít người biết mà có đúng một láng
giềng rất giống được xếp ngang phim nổi tiếng có ba mươi láng giềng.

Hai công thức trả lời hai câu hỏi khác nhau: `weighted` cho MAE/RMSE, `sum` cho
top-K.

### 3. Chọn số láng giềng k trên validation

| k | ItemKNN val | UserKNN val | ItemKNN test | UserKNN test | ItemKNN coverage@10 |
|---|---|---|---|---|---|
| 5 | 0.3710 | 0.2977 | 0.3699 | 0.2950 | 0.0951 |
| 10 | 0.4081 | 0.3692 | 0.4291 | 0.3538 | 0.0880 |
| 20 | 0.4467 | 0.4225 | 0.4564 | 0.3999 | 0.0801 |
| 40 | 0.4648 | **0.4441** | 0.4688 | 0.4214 | 0.0773 |
| 80 | **0.4768** | 0.4334 | 0.4761 | 0.4272 | 0.0749 |
| 160 | 0.4751 | 0.4182 | 0.4761 | 0.4221 | 0.0735 |

nDCG@10. Validation chọn k = 80 cho ItemKNN, k = 40 cho UserKNN. Hai số này nằm ở
`K_ITEM`, `K_USER` trong [recsys/models/knn.py](recsys/models/knn.py), và mọi script
lấy k từ đó. Chạy lại mà validation chọn khác thì `exp_knn.py` in cảnh báo.

Đường cong phẳng từ k = 40 trở đi: trên test, k = 80 hơn k = 40 +0.007 nDCG, CI
[−0.006, +0.021], tức là nhiễu. Khớp với slide L2 tr.22 (thực nghiệm trên
MovieLens gợi ý 20-50).

Coverage **giảm đều** khi k tăng: nhiều láng giềng thì nhiều đồng thuận, tức
phim phổ biến. Cùng một đánh đổi, lần thứ ba xuất hiện.

## Đo thêm: toàn danh mục, AUC, cold-start

```bash
python scripts/exp_giao_thuc.py   # ~3 phút, ghi reports/exp_2_*.csv
```

Checklist 2.4, 2.5, 2.8. Không đổi giao thức chung, chỉ đo thêm. Số đầy đủ kèm CI
và kết luận: [docs/so-lieu-dot-2.md](docs/so-lieu-dot-2.md).

**Toàn danh mục (2.4).** Xếp phim đúng giữa ~9.700 phim user chưa xem, thay vì 99
phim lấy mẫu đều. Chỉ 24-38/610 user trúng top-10, nên CI rộng. ItemKNN hơn
Popularity +0.014 nDCG@10, CI [+0.003, +0.026]: có ý nghĩa nhưng cận dưới sát 0.
UserKNN − Popularity và ItemKNN − UserKNN về nDCG@10 là nhiễu. Báo cáo phải ghi
giao thức cạnh câu "+33%", và bảng cuối kỳ nên chạy thêm trên ml-1m (6.040 user).

**Cold-start (2.8).** Cắt lịch sử train của từng 1/4 số user xuống n tương tác gần
nhất, đo nDCG@10 trên đủ 610 user:

| lịch sử | Popularity | ItemKNN | UserKNN |
|---|---|---|---|
| 1 | **0.361** | 0.208 | 0.044 |
| 3 | **0.361** | 0.305 | **0.359** |
| 5 | 0.360 | 0.352 | **0.405** |
| 10 | 0.361 | **0.400** | **0.413** |
| đầy đủ | 0.359 | **0.476** | 0.421 |

Ô in đậm là tốt nhất hoặc hòa với tốt nhất (CI ghép cặp chứa 0). UserKNN hơn
ItemKNN có ý nghĩa ở 3 và 5 tương tác, hòa ở 10, thua ở lịch sử đầy đủ. Với 1
tương tác, UserKNN không có láng giềng nào (Pearson của một rating là không xác
định), và Popularity thắng cả hai. Đây là căn cứ đo được cho lai ghép kiểu chuyển
ở B7. MF ở B3 và BPR ở B4 làm căn cứ này không còn đứng được, xem phần dưới.

## Bậc B3: phân rã ma trận

```bash
python scripts/exp_mf.py   # ~15-20 phút, ghi reports/exp_mf_*.csv
```

Cài trong [recsys/models/mf.py](recsys/models/mf.py) bằng numpy, huấn luyện bằng ALS. Số
đầy đủ kèm CI và kết luận: [docs/so-lieu-b3.md](docs/so-lieu-b3.md).

- **Hai chế độ.** `implicit` học việc user có xem hay không, dùng cho top-K, là mặc
  định. `explicit` học rating có bias, dùng cho MAE/RMSE. Siêu tham số chọn trên
  validation: implicit f = 32, reg = 10, alpha = 10; explicit f = 32, reg = 30.
- **MF implicit nhỉnh hơn ItemKNN, sát ngưỡng.** nDCG@10 +0.024, CI [+0.002, +0.046]
  khi lấy mẫu. Trên toàn danh mục chênh lệch này là nhiễu. MF hơn Popularity có ý
  nghĩa ở cả hai giao thức.
- **MF phủ danh mục rộng hơn KNN** (coverage@10 10,1% so với 7,5%), dù chính xác hơn.
- **MF explicit thua Popularity** khi xếp hạng (nDCG@10 −0.100 khi lấy mẫu). Cùng bài
  học với `weighted` của KNN.
- **Cold-start:** MF hơn cả hai KNN ở mọi độ dài lịch sử, kể cả 1 tương tác. Ở 1
  tương tác, MF hòa Popularity (+0.029, CI chứa 0).
- **Câu giải thích là phân rã chính xác.** Điểm của phim gợi ý bằng đúng tổng đóng góp
  của các phim user đã xem (`MF.contributions()`). `explain()` nêu ba phim đóng góp
  nhiều nhất.

## Bậc B4: BPR-MF

```bash
python scripts/exp_bpr.py   # ~20-30 phút, ghi reports/exp_bpr_*.csv
```

Cài trong [recsys/models/bpr.py](recsys/models/bpr.py) bằng numpy, học bằng SGD theo lô.
Cùng dạng điểm với MF, nhưng chỉ học thứ tự: phim đã xem phải xếp trên phim chưa xem. Số
đầy đủ kèm CI và kết luận: [docs/so-lieu-b4.md](docs/so-lieu-b4.md).

- **Siêu tham số chọn trên validation:** f = 64, lr = 0.1, reg = 0.01, 80 epoch.
- **BPR và MF implicit hòa về nDCG@10 và recall@10**, ở cả hai giao thức.
- **BPR hơn MF về AUC:** +0.025, CI [+0.014, +0.037]. Khớp lý thuyết, vì mục tiêu của
  BPR là dạng liên tục của AUC.
- **Bias của BPR gần như là độ phổ biến** (Spearman 0.949 với số lượt xem). Dù vậy top-10
  của BPR không thiên về phim phổ biến hơn MF.
- **Cold-start:** với 1 tương tác, BPR hơn cả MF (+0.047) và Popularity (+0.076), đều có
  ý nghĩa. Từ 3 tương tác trở lên BPR hòa MF. Như vậy chưa có mức lịch sử nào mà chuyển
  sang Popularity giúp được. B7 cần căn cứ khác: phim mới chưa ai xem, xem B5.
- **Câu giải thích là hậu kiểm**, nêu phim đã xem có vector gần phim gợi ý nhất. Không
  phân rã điểm chính xác được như B3.

## Bậc B5: gợi ý theo nội dung

```bash
python scripts/fetch_tmdb.py      # cần cache TMDB trước, xem phần dưới
python scripts/exp_content.py     # ~15-20 phút, ghi reports/exp_content_*.csv
```

Cài trong [recsys/models/content.py](recsys/models/content.py). Mỗi phim là một vector TF-IDF
từ metadata, hồ sơ user là tổng các vector phim đã xem, điểm là cosine. Số đầy đủ kèm CI và
kết luận: [docs/so-lieu-b5.md](docs/so-lieu-b5.md).

- **Siêu tham số chọn trên validation:** thể loại, thập niên, đạo diễn, diễn viên, từ khóa;
  trọng số `binary`; half_life 3 (ưu tiên phim xem gần nhất); trọng số thể loại 1, đạo diễn 4.
  Thêm tóm tắt phim hay trọng số theo rating đều làm kết quả kém đi.
- **Khi lấy mẫu, B5 thua cả Popularity** (nDCG@10 −0.103, có ý nghĩa). Trên toàn danh mục,
  chênh lệch nDCG@10 với Popularity, ItemKNN, BPR đều là nhiễu. Lý do: B5 không biết độ phổ
  biến, mà phim âm lấy mẫu phần lớn là phim ít người xem.
- **B5 phủ 27,9% danh mục**, gấp 3 lần BPR. Phim nó gợi ý có trung vị 7 lượt xem, của BPR là 108.
- **Phim mới:** khi phim test bị xóa khỏi train, Popularity và BPR về nDCG@10 bằng 0, B5 vẫn
  được 0.252. Đây là căn cứ cho B7: lai ghép kiểu chuyển theo phim, không theo user.
- **User mới:** B5 thua Popularity và BPR ở mọi mức lịch sử, nên không dùng cho user mới.
- **Câu giải thích là phân rã chính xác**, theo từng đặc trưng (`shared_features()`) và từng
  phim đã xem (`contributions()`).
- `recommend()` chỉ duyệt phim có trong train, nên B5 chưa đưa phim mới lên giao diện. B7 mở rộng
  danh mục ứng viên, xem [phần B7](#bậc-b7-lai-ghép).
- Chưa có cache TMDB thì B5 chỉ dùng thể loại và thập niên. nDCG@10 validation khi đó là 0.2035
  thay vì 0.2571.

## Bậc B6: luật kết hợp theo phiên

```bash
python scripts/exp_rules.py       # ~70 phút, ghi reports/exp_rules_*.csv
```

Cài trong [recsys/models/rules.py](recsys/models/rules.py), khai phá bằng `mlxtend`. Phiên là
chuỗi rating của một user, hai rating liền nhau cách không quá 1800 giây. Luật {a} → {j} khai
phá trên các phiên. Điểm của phim j là confidence lớn nhất của các luật có a trong 10 phim gần
nhất. Số đầy đủ kèm CI và kết luận: [docs/so-lieu-b6.md](docs/so-lieu-b6.md).

- **Siêu tham số chọn trên validation:** khoảng nghỉ 1800 giây, min_support 0,0005, max_len 2,
  ngữ cảnh 10 phim gần nhất, confidence, lấy max. nDCG@10 validation 0.4995, BPR được 0.4928.
- **B6 hơn Popularity** +0.119 nDCG@10 khi lấy mẫu và +0.024 trên toàn danh mục, đều có ý nghĩa.
  Với ItemKNN và BPR, chênh lệch nDCG@10 ở cả hai giao thức là nhiễu.
- **556/610 phim test nằm cùng phiên với rating cuối trong train.** Trên MovieLens, phiên là
  một buổi chấm điểm hàng loạt, không phải một buổi xem phim.
- **Độ phủ:** top-10 của B6 có 1.037 phim khác nhau, BPR có 884. Phim của B6 phổ biến hơn:
  59,9% thuộc top-100, của BPR là 48,5%.
- **User mới:** B6 hơn Popularity từ 1 tương tác, thua BPR có ý nghĩa ở 1, 5, 10 tương tác.
- **Câu giải thích** là luật mạnh nhất, kèm confidence và số phiên chứa cả hai phim.
- Mỗi lần `fit()` khai phá lại, mất 218 giây ở min_support 0,0005. `run_eval.py` vì vậy chạy
  lâu hơn trước.

## Bậc B7: lai ghép

```bash
python scripts/exp_hybrid.py      # cần cache TMDB, ~45 phút, ghi reports/exp_hybrid_*.csv
```

Cài trong [recsys/models/hybrid.py](recsys/models/hybrid.py), ghép B0, B4, B5 theo ba nhánh. User
chưa có lịch sử dùng Popularity. Phim BPR không chấm được dùng B5. Phần còn lại trộn
`(1 − α_u) · BPR + α_u · B5` trên thang phân vị của từng user, với `α_u = α · n / (n + c)` tăng
theo số tương tác n. Số đầy đủ kèm CI và kết luận: [docs/so-lieu-b7.md](docs/so-lieu-b7.md).

- **Siêu tham số chọn trên validation, hai bước.** Bước 1 chọn chuẩn hóa theo phân vị, α = 0.1.
  Bước 2 chọn dạng α_u trên validation có cắt lịch sử xuống 1, 3, 5, 10 tương tác: c = 30.
- **Khi lấy mẫu, B7 hơn BPR** +0.033 nDCG@10, CI [+0.018, +0.047], và hơn MF +0.021, CI
  [+0.001, +0.042]. Trên toàn danh mục, cả hai chênh lệch là nhiễu.
- **Phần hơn AUC chủ yếu đến từ 23 user có phim test mới.** Trên 610 user, B7 hơn BPR +0.031 AUC.
  Bỏ 23 user đó, chỉ còn +0.006.
- **Phim mới lên được giao diện.** 26/41 phim chưa ai xem lọt vào top-10 của ít nhất một user.
  Chúng chiếm 2,07% số chỗ trong top-10, gấp khoảng 5 lần tỉ lệ 0,4% trong danh mục.
- **User mới:** ở 0 tương tác, nhánh Popularity hơn BPR +0.317 nDCG@10. So với α chung, α theo
  lịch sử tăng +0.020 ở 1 tương tác (có ý nghĩa) và không đổi có ý nghĩa ở lịch sử đầy đủ. B7 vẫn
  kém BPR −0.010 ở 1 tương tác. Phần này đến từ phim mới chen vào 99 phim âm, không từ α.
- **Trên validation có cắt lịch sử, BPR một mình có trung bình nDCG@10 cao hơn mọi cấu hình B7**
  (0.4831 so với 0.4816). Lợi thế của B7 chỉ có ở user lịch sử đầy đủ.
- **Câu giải thích** lấy từ nhánh đóng góp nhiều hơn vào điểm. Phim mới dùng câu của B5, kèm
  tiền tố "Phim moi, chua ai danh gia".

## Bậc B8: LightGCN

```bash
python scripts/exp_lightgcn.py    # ~1,5-2 giờ, ghi reports/exp_lightgcn_*.csv
```

Cài trong [recsys/models/lightgcn.py](recsys/models/lightgcn.py) bằng numpy và scipy, không cần
PyTorch. Cùng hàm mất mát và dạng điểm với BPR, thêm K lớp lan truyền trên đồ thị user-phim. Lan
truyền là tuyến tính và ma trận kề chuẩn hóa đối xứng, nên gradient tính bằng chính phép lan
truyền. Số đầy đủ kèm CI và kết luận: [docs/so-lieu-b8.md](docs/so-lieu-b8.md).

- **Siêu tham số chọn trên validation:** K = 4, lr = 0.01, reg = 1e-4, 300 epoch, lô 8.192, Adam.
  nDCG@10 validation 0.5077. K = 4 và 300 epoch nằm ở biên lưới, chưa thử lớn hơn.
- **Khi lấy mẫu, B8 hơn BPR +0.022 nDCG@10**, CI [+0.006, +0.038], sát ngưỡng. Với K = 3 chênh
  lệch này là nhiễu. Trên toàn danh mục, B8 hòa BPR, MF và ItemKNN.
- **AUC bằng BPR** (0.8858), vì hai mô hình cùng tối ưu BPR-Opt. Cả hai hơn MF khoảng 0.025.
- **So với K = 0, nDCG@10 tăng +0.081.** Phần lớn con số này là khoảng cách giữa K = 0 và B4
  (−0.059): K = 0 không có bias và học quá lâu, đỉnh validation ở 100 epoch. So với B4 đã chỉnh
  riêng, đồ thị chỉ thêm +0.022.
- **Độ phủ:** top-10 có 1.023 phim khác nhau, nhiều nhất trong các mô hình lọc cộng tác.
- **User mới:** ở 1 tương tác, B8 kém BPR −0.029, có ý nghĩa. Từ 3 tương tác trở lên hai mô hình hòa.
- **Câu giải thích là hậu kiểm**, kế thừa từ BPR.
- Fit mất khoảng 380 giây, gấp 12 lần BPR. `run_eval.py` chạy lâu hơn trước khoảng 6 phút.

## Metadata TMDB

```bash
python scripts/fetch_tmdb.py --limit 20   # thử nhanh 20 phim trước
python scripts/fetch_tmdb.py              # tải hết (~40 phút, chạy một lần)
python scripts/fetch_tmdb.py --report     # xem cache đang phủ được bao nhiêu
```

Cache ghi kiểu JSONL ở `data/cache/tmdb_movies.jsonl`, **nối tiếp được**:
đứt mạng ở phim thứ 5.000 thì chạy lại chỉ tải 4.724 phim còn lại.

**Chưa có TMDB key vẫn chạy được.** `movies.csv` của MovieLens đã có sẵn
thể loại, đủ để bắt đầu bậc B5 ngay hôm nay — 9.708/9.742 phim có ít nhất
một thể loại. Có cache TMDB thì bổ sung thêm diễn viên, đạo diễn, từ khóa.

Cột `has_tmdb` nói rõ từng phim có metadata TMDB hay không. Không có cột này
thì phim *thiếu* dữ liệu trông y hệt phim có dữ liệu *rỗng*, và B5 sẽ báo
"phim này không giống phim nào" cho một thứ thật ra là thiếu metadata.

### Một sửa lỗi có chủ ý so với repo tham khảo

Repo gốc đưa thẳng tên người vào `CountVectorizer`, nên `James Cameron` bị
tách thành `james` + `cameron`. Hậu quả: mọi phim có diễn viên tên Chris,
John hay Lee đều tương đồng giả với nhau. `tmdb._token()` nối tên lại thành
một token — `chris_evans` và `chris_pratt` là hai token khác hẳn nhau.

Đã đo ở `scripts/exp_content.py` (mục 3): tách tên như repo gốc làm nDCG@10 của B5
giảm từ 0.256 xuống 0.200. Chênh lệch +0.057, CI 95% [+0.040, +0.074].

## Cấu trúc

```
recsys/
  base.py              hợp đồng chung — đọc file này trước
  data/movielens.py    tải + nạp dữ liệu, nối sang TMDB qua links.csv
  data/tmdb.py         metadata TMDB có cache; chuỗi đặc trưng cho B5
  data/split.py        leave-one-out theo thời gian, validation, cắt lịch sử
  eval/metrics.py      P@K, R@K, MAP, MRR, nDCG, AUC, coverage
  eval/runner.py       evaluate() một mô hình, compare() nhiều mô hình
  eval/stats.py        khoảng tin cậy bootstrap theo user
  eval/provenance.py   ghi CSV kèm .meta.json: seed, git commit, phiên bản thư viện
  models/baselines.py  B0 — Popularity, Random
  models/knn.py        B1 — Item-KNN · B2 — User-KNN (Pearson), k đã chọn
  models/mf.py         B3 — phân rã ma trận ALS, implicit và explicit
  models/bpr.py        B4 — BPR-MF, xếp hạng theo cặp, SGD theo lô
  models/content.py    B5 — nội dung, hồ sơ user TF-IDF từ metadata
  models/rules.py      B6 — luật kết hợp theo phiên, khai phá bằng mlxtend
  models/hybrid.py     B7 — lai ghép: Popularity, BPR, nội dung theo nhánh
  models/lightgcn.py   B8 — LightGCN, lan truyền trên đồ thị user-phim, numpy thuần
scripts/run_eval.py    CLI chạy so sánh
scripts/fetch_tmdb.py  tải metadata TMDB về cache
scripts/exp_knn.py     ba thí nghiệm của bậc B1/B2, chọn k trên validation
scripts/exp_giao_thuc.py  toàn danh mục, tách AUC, cold-start (checklist 2.4/2.5/2.8)
scripts/exp_mf.py      thí nghiệm B3: chọn siêu tham số, bảng test, cold-start
scripts/exp_bpr.py     thí nghiệm B4: như B3, thêm thiên lệch phổ biến
scripts/exp_content.py thí nghiệm B5: như B4, thêm phim mới, nối tên người
scripts/exp_rules.py   thí nghiệm B6: cắt phiên, min_support, cách chấm điểm, user mới
scripts/exp_hybrid.py  thí nghiệm B7: chọn α, phim mới, user mới kể cả 0 tương tác
scripts/exp_lightgcn.py  thí nghiệm B8: chọn K, đóng góp của đồ thị, phổ biến, user mới
scripts/so_lieu_slide.py  mọi số cho slide, kèm CI 95% (in Markdown)
tests/                 chốt độ đo, phép chia, runner, CI, metadata, KNN, MF, BPR, B5-B8 (264 test)
docs/checklist.md      việc cần sửa, chia đợt, có người phụ trách
docs/so-lieu-slide.md  số liệu và câu cần sửa cho từng slide
docs/so-lieu-dot-2.md  số liệu và kết luận của 2.4, 2.5, 2.8
docs/so-lieu-b3.md     số liệu và kết luận của B3
docs/so-lieu-b4.md     số liệu và kết luận của B4
docs/so-lieu-b5.md     số liệu và kết luận của B5
docs/so-lieu-b6.md     số liệu và kết luận của B6
docs/so-lieu-b7.md     số liệu và kết luận của B7
docs/so-lieu-b8.md     số liệu và kết luận của B8
```

## Mã nguồn và dữ liệu tái sử dụng

| Thành phần | Nguồn |
|---|---|
| MovieLens | GroupLens Research |
| TMDB API | The Movie Database |
| `pandas`, `scikit-learn`, `PyTorch`, `implicit`, `mlxtend`, `Flask` | thư viện mã nguồn mở |
| `End-to-End-Movie-Recommendation-System` | [KalyanM45](https://github.com/KalyanM45/End-to-End-Movie-Recommendation-System), GPL v3 — dùng làm **baseline được đo đạc**, không phải nền tảng |

Repo tham khảo nằm ngoài dự án này và không được sửa. Nó cài đặt đúng một
kỹ thuật (tương đồng cosine giữa phim theo diễn viên/đạo diễn/thể loại),
không có người dùng, không có lịch sử xem, không có đánh giá chất lượng —
tương ứng bậc B5 và còn thiếu phần hồ sơ người dùng.

## Khóa API

Chép `.env.example` thành `.env` rồi điền `TMDB_API_KEY`. File `.env` đã nằm
trong `.gitignore`. Không nhúng khóa vào JavaScript như repo tham khảo — ai
mở trang web cũng đọc được và dùng hết quota của nhóm.
