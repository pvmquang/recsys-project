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

pytest -q                       # 181 test, không gọi mạng
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
| **B3-MF** implicit (f=32) | **0.7279** | **0.5003** | **0.4386** | 0.8605 | 0.1006 |

Số láng giềng k và siêu tham số của MF chọn trên tập validation (xem
[Giao thức](#giao-thức-đánh-giá)). Dòng B3 lấy từ `scripts/exp_mf.py`, xem
[phần B3](#bậc-b3-phân-rã-ma-trận).
AUC tính từ điểm, cặp bằng điểm tính 0.5. Coverage là coverage@10, cùng k với
các cột bên cạnh.

**Cả bảng này đo trên 1 phim đúng + 99 phim âm lấy mẫu.** Trên toàn danh mục,
chỉ còn ItemKNN và MF hơn Popularity có ý nghĩa, và đều sát ngưỡng. MF với ItemKNN
thì hòa. Xem
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
ở B7. MF ở B3 làm căn cứ này yếu đi, xem phần dưới.

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

Đây là cải tiến nhỏ, đo được bằng số, giải thích được bằng một câu — nên
báo cáo nên đo **cả hai cách** rồi trình bày chênh lệch.

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
scripts/run_eval.py    CLI chạy so sánh
scripts/fetch_tmdb.py  tải metadata TMDB về cache
scripts/exp_knn.py     ba thí nghiệm của bậc B1/B2, chọn k trên validation
scripts/exp_giao_thuc.py  toàn danh mục, tách AUC, cold-start (checklist 2.4/2.5/2.8)
scripts/exp_mf.py      thí nghiệm B3: chọn siêu tham số, bảng test, cold-start
scripts/so_lieu_slide.py  mọi số cho slide, kèm CI 95% (in Markdown)
tests/                 chốt độ đo, phép chia, runner, CI, metadata, KNN, MF (181 test)
docs/checklist.md      việc cần sửa, chia đợt, có người phụ trách
docs/so-lieu-slide.md  số liệu và câu cần sửa cho từng slide
docs/so-lieu-dot-2.md  số liệu và kết luận của 2.4, 2.5, 2.8
docs/so-lieu-b3.md     số liệu và kết luận của B3
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
