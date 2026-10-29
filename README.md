# Hệ gợi ý phim lai ghép — đồ án IT4613

Nghiên cứu so sánh các kỹ thuật gợi ý trên MovieLens + TMDB, từ baseline
đến mô hình lai ghép, đo trên cùng một giao thức đánh giá.

## Chạy thử

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt

pytest tests/ -q                # 123 test
python scripts/run_eval.py      # tự tải MovieLens ở lần chạy đầu
```

## Kết quả hiện có (ml-latest-small, seed 42, k=40)

```bash
python scripts/so_lieu_slide.py   # mọi số dưới đây, kèm CI 95%
```

| Mô hình | recall@10 | ndcg@10 | MRR | AUC | coverage@10 |
|---|---|---|---|---|---|
| B0-Random | 0.1180 | 0.0538 | 0.0561 | 0.4956 | 0.4670 |
| B0-Popularity | 0.6033 | 0.3603 | 0.3015 | 0.8319 | 0.0125 |
| **B1-ItemKNN** | **0.6754** | **0.4744** | **0.4203** | 0.8205 | 0.0768 |
| B2-UserKNN | 0.6508 | 0.4238 | 0.3662 | 0.8412 | 0.0288 |

Coverage là coverage@10 — cùng k với các cột bên cạnh. (Bản trước ghi
"coverage" nhưng thực ra là @20.)

**Kiểm tra bằng Random không phát hiện lỗi khung đo.** Qua 10 seed: recall@10
= 0.101 ± 0.012, AUC = 0.503 ± 0.012 — khớp kỳ vọng lý thuyết 0.10 và 0.50.
Nếu sau này sửa gì mà hai số này lệch đi, lỗi nằm ở khung đo chứ không ở mô hình.
(Bản trước gieo seed bằng `seed + user_id` nên mọi seed gần như trùng nhau, và
một lần rút xui — AUC 0.5258 — lặp lại ở mọi seed.)

**Cá nhân hóa thắng, và thắng trên CẢ HAI mặt.** ItemKNN hơn Popularity
+0.114 nDCG@10 (+32%, CI 95% [+0.091, +0.135]) *và* phủ danh mục gấp 6 lần
(7,7% ↔ 1,25%). Ở đây không có đánh đổi — đó là câu trả lời cho thanh ngang
mà Popularity đặt ra. ItemKNN cũng hơn UserKNN +0.051 nDCG@10, CI [+0.028, +0.072].

**AUC: chưa kết luận được.** Mọi chênh lệch đo bằng bootstrap ghép cặp theo user:

| so sánh AUC | chênh lệch | CI 95% | |
|---|---|---|---|
| UserKNN − Popularity | +0.009 | [−0.006, +0.024] | nhiễu |
| ItemKNN − Popularity | −0.011 | [−0.030, +0.008] | nhiễu |
| ItemKNN − UserKNN | −0.021 | [−0.039, −0.002] | có ý nghĩa, nhưng xem dưới |

Riêng cặp cuối: 88% ứng viên của ItemKNN và 72% của UserKNN có điểm `-inf`
(không láng giềng nào), và thứ tự giữa chúng do xáo ngẫu nhiên quyết định. AUC
của KNN vì vậy chủ yếu đo *"phim đúng có láng giềng không"*, không đo chất lượng
xếp hạng. Cần AUC xử lý thế hòa (checklist 2.5) trước khi viết câu nào về AUC.

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

## Giao thức đánh giá

Leave-one-out **theo thời gian**: tương tác cuối cùng của mỗi user vào tập
kiểm tra, chấm điểm trên 1 phim đúng trộn với 99 phim lấy ngẫu nhiên.

Không dùng chia ngẫu nhiên — nó cho mô hình nhìn thấy tương lai của chính
user đó, làm mọi chỉ số cao lên một cách giả tạo.

Hạn chế còn lại, phải nêu trong báo cáo:
- Train vẫn chứa tương tác về sau của user **khác** (ví dụ số đếm của
  Popularity). Giao thức chỉ chặn tương lai *của chính user đó*.
- 94/610 user có nhiều phim trùng timestamp cuối (nhiều nhất 34 phim cùng một
  giây); khi đó phim test do phá hòa theo `item_id` quyết định.
- 23/610 phim test chưa từng xuất hiện trong train — không mô hình lọc cộng
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

### 1. Theo nhóm user — ItemKNN thắng ở cả ba nhóm

nDCG@10. Nhóm chia theo **phân vị** số tương tác trong train, mỗi nhóm ~200 user:

| nhóm (số tương tác) | số user | Popularity | ItemKNN | UserKNN | ItemKNN − UserKNN, CI 95% |
|---|---|---|---|---|---|
| 19–44 | 209 | 0.4241 | **0.5563** | 0.5071 | +0.049 [+0.007, +0.090] |
| 45–126 | 198 | 0.3955 | **0.5287** | 0.4722 | +0.056 [+0.020, +0.094] |
| 127–2697 | 203 | 0.2603 | **0.3371** | 0.2908 | +0.046 [+0.019, +0.076] |

- **ItemKNN thắng UserKNN ở cả ba nhóm, chênh lệch có ý nghĩa.** Popularity
  thấp nhất ở mọi nhóm.
- **Bản trước kết luận ngược lại** — "nhóm 5–20 thì UserKNN tốt hơn, Popularity
  thắng recall" — từ một nhóm thực chất là 29 user có 19–20 tương tác, chênh
  lệch nằm trong nhiễu (CI [−0.090, +0.159]). Kết luận đó **không** còn là căn
  cứ cho B7.
- **Mọi mô hình đều kém nhất ở nhóm nhiều lịch sử nhất.** Độ khó của user nhiều
  lịch sử là tính chất của dữ liệu, không phải điểm yếu riêng của Popularity.
- ⚠️ **Đây không phải phân tích cold-start**: nhóm thưa nhất vẫn có ≥ 19 tương
  tác. Muốn đo cold-start phải cắt bớt lịch sử train (checklist 2.8).

### 2. Hai chế độ chấm điểm

| | recall@10 | ndcg@10 | MRR | coverage@10 |
|---|---|---|---|---|
| ItemKNN `sum` | 0.6754 | **0.4744** | 0.4203 | 0.0768 |
| ItemKNN `weighted` | 0.6361 | 0.3354 | 0.2537 | **0.2430** |
| UserKNN `sum` | 0.6508 | **0.4238** | 0.3662 | 0.0288 |
| UserKNN `weighted` | 0.4672 | 0.2111 | 0.1583 | **0.1159** |

`weighted` là công thức **dự đoán rating** ở slide L2 tr.17/22. Nó tụt 29–50%
nDCG nhưng phủ danh mục gấp **3–4 lần**. Lý do: phép chia cho `Σ sim` xoá mất
tín hiệu *"có bao nhiêu bằng chứng"*, nên một phim ít người biết mà có đúng
một láng giềng rất giống được xếp ngang phim nổi tiếng có ba mươi láng giềng.

Đây không phải lỗi — **hai công thức trả lời hai câu hỏi khác nhau**:
`weighted` cho MAE/RMSE, `sum` cho top-K.

### 3. Quét số láng giềng k

| k | ItemKNN nDCG@10 | UserKNN nDCG@10 | ItemKNN coverage@10 |
|---|---|---|---|
| 5 | 0.3633 | 0.2944 | 0.0958 |
| 20 | 0.4654 | 0.4036 | 0.0800 |
| 40 | 0.4744 | 0.4238 | 0.0768 |
| 80 | 0.4752 | 0.4269 | 0.0751 |
| 160 | 0.4738 | 0.4218 | 0.0735 |

Đường cong **phẳng từ k = 40 trở đi** — k=80 hơn k=40 đúng +0.0008 nDCG, CI
[−0.012, +0.014], tức là nhiễu.

⚠️ k đang được chọn ngay trên **tập kiểm tra** vì chưa có tập validation
(checklist 2.1). Đọc bảng này như một đường cong, không phải một lựa chọn.
Khớp với slide L2 tr.22 (thực nghiệm trên MovieLens gợi ý 20–50), và đây là
lần kiểm lại con số đó bằng chính dữ liệu của nhóm.

⚠️ Coverage **giảm đều** khi k tăng: nhiều láng giềng ⇒ nhiều đồng thuận ⇒
phim phổ biến. Cùng một đánh đổi, lần thứ ba xuất hiện.

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
  data/split.py        leave-one-out theo thời gian + lấy mẫu item âm
  eval/metrics.py      P@K, R@K, MAP, MRR, nDCG, AUC, coverage
  eval/runner.py       evaluate() một mô hình, compare() nhiều mô hình
  eval/stats.py        khoảng tin cậy bootstrap theo user
  models/baselines.py  B0 — Popularity, Random
  models/knn.py        B1 — Item-KNN · B2 — User-KNN (Pearson)
scripts/run_eval.py    CLI chạy so sánh
scripts/fetch_tmdb.py  tải metadata TMDB về cache
scripts/exp_knn.py     ba thí nghiệm của bậc B1/B2
scripts/so_lieu_slide.py  mọi số cho slide, kèm CI 95% (in Markdown)
tests/                 chốt độ đo, phép chia, runner, CI, metadata, KNN (123 test)
docs/checklist.md      việc cần sửa, chia đợt, có người phụ trách
docs/so-lieu-slide.md  số liệu và câu cần sửa cho từng slide
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
