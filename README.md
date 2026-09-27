# Semantic Search API (FastAPI + PostgreSQL + pgvector)

Dự án **Semantic Search API** hoàn chỉnh kết hợp sức mạnh của **FastAPI**, **PostgreSQL** cùng extension **pgvector**, và mô hình nhúng văn bản đa ngôn ngữ **Sentence-Transformers** (`paraphrase-multilingual-MiniLM-L12-v2`). Hệ thống cho phép tìm kiếm ngữ nghĩa (Semantic Search) vượt trội so với tìm kiếm từ khóa truyền thống (Keyword Search), đồng thời hỗ trợ tìm kiếm xuyên ngôn ngữ (**Cross-Lingual Search: query tiếng Việt trên tài liệu tiếng Anh**).

---

## 🏛 Sơ đồ kiến trúc hệ thống

```text
+-----------------------------------------------------------------------------------+
|                                  CLIENT REQUEST                                   |
|               (Swagger UI / cURL / Frontend / External Service)                   |
+----------------------------------------+------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                                FASTAPI APPLICATION                                |
|  - POST /documents   (Ingestion & Dynamic Chunking)                               |
|  - GET  /search      (Cosine Similarity Query)                                    |
|  - GET  /health      (Healthcheck & Pool Status)                                  |
+-------------------+-------------------------------------------+-------------------+
                    |                                           |
                    v                                           v
    +-------------------------------+           +-------------------------------+
    |       CHUNKING MODULE         |           |       EMBEDDING MODULE        |
    |      (app/chunking.py)        |           |      (app/embedding.py)       |
    |  - tiktoken (cl100k_base)     |           |  - SentenceTransformer        |
    |  - Configurable stride/overlap|           |    (paraphrase-multilingual-  |
    |                               |           |     MiniLM-L12-v2: 384 dim)   |
    +---------------+---------------+           +---------------+---------------+
                    |                                           |
                    +---------------------+---------------------+
                                          | (Chunks + Vectors)
                                          v
+-----------------------------------------------------------------------------------+
|                        POSTGRESQL 16 DATABASE (pgvector)                          |
|  Table: documents                                                                 |
|   - id: SERIAL PRIMARY KEY                                                        |
|   - parent_title: TEXT                                                            |
|   - chunk_index: INT                                                              |
|   - content: TEXT                                                                 |
|   - embedding: VECTOR(384)                                                        |
|   - so_question_id: BIGINT (Stack Overflow Question ID)                          |
|   - content_tsv: tsvector (Full-Text Search)                                      |
|                                                                                   |
|  Indexes:                                                                         |
|   - HNSW Index: idx_documents_hnsw USING hnsw (embedding vector_cosine_ops)       |
|   - GIN Index:  idx_documents_content_tsv USING gin(content_tsv)                  |
+-----------------------------------------------------------------------------------+
```

---

## 📁 Cấu trúc thư mục

```text
semantic-search/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI entry point, lifecycle và các route API
│   ├── database.py        # Connection pool asyncpg và tự động khởi tạo schema
│   ├── models.py          # Domain data models
│   ├── schemas.py         # Pydantic schemas (DocumentIn, DocumentOut, SearchResult)
│   ├── embedding.py       # Module sinh vector nhúng Sentence-Transformers (384d)
│   ├── chunking.py        # Module chia nhỏ văn bản bằng tiktoken theo token
│   └── search.py          # Logic insert document và truy vấn similarity pgvector
├── scripts/
│   ├── prepare_data.py    # Download dataset Kaggle, làm sạch HTML và lưu so_question_id
│   ├── build_eval_set.py  # Tạo bộ 30 câu truy vấn đánh giá tiếng Anh
│   ├── build_eval_set_vi.py # Tạo bộ 30 câu truy vấn đánh giá tiếng Việt song song
│   ├── seed.py            # Nạp dữ liệu vào database (--limit, --clear)
│   ├── reseed.py          # Xoá và nạp lại toàn bộ dữ liệu với embedding model mới
│   ├── add_so_question_id_column.py      # Migration thêm và backfill cột so_question_id
│   ├── experiment_chunking.py            # Experiment 1: Đánh giá chiến lược chunking
│   ├── experiment_keyword_vs_semantic.py # Experiment 2: So sánh Keyword vs Semantic
│   ├── experiment_index_latency.py       # Experiment 3: Benchmark latency HNSW
│   └── experiment_cross_lingual.py       # Experiment Cross-Lingual: Tiếng Anh vs Tiếng Việt
├── data/
│   ├── documents_dev.json                # Tập dữ liệu 3,000 documents dev
│   ├── documents_full.json               # Tập dữ liệu 14,997 documents HQ full
│   ├── eval_queries.json                 # 30 câu truy vấn đánh giá tiếng Anh
│   ├── eval_queries_vi.json              # 30 câu truy vấn đánh giá tiếng Việt
│   ├── results_chunking.json             # Kết quả Experiment 1
│   ├── results_keyword_vs_semantic.json  # Kết quả Experiment 2
│   ├── results_index_latency.json        # Kết quả Experiment 3
│   ├── results_cross_lingual.json        # Kết quả Experiment Cross-Lingual
│   └── latency_chart.png                 # Biểu đồ so sánh latency Flat vs HNSW
├── tests/
│   ├── test_health.py     # Test health endpoint
│   ├── test_chunking.py   # Unit test logic chunking và overlap
│   ├── test_embedding.py  # Unit test kích thước vector nhúng (384 dims)
│   └── test_api.py        # Integration test endpoints /documents và /search
├── docker-compose.yml     # Khởi chạy PostgreSQL 16 + pgvector container
├── requirements.txt       # Danh sách dependencies
├── .env.example           # Cấu hình mẫu biến môi trường
└── README.md
```

---

## 🚀 Hướng dẫn cài đặt và chạy (Setup)

### 1. Khởi động PostgreSQL với pgvector

Khởi chạy container PostgreSQL 16 ở chế độ nền:

```bash
docker compose up -d
```

Kiểm tra trạng thái container:
```bash
docker ps --filter "name=semantic_search_postgres"
```

### 2. Thiết lập Virtual Environment (venv)

Trên **Windows (PowerShell)**:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Trên **Linux / macOS**:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

### 4. Cấu hình biến môi trường

```bash
cp .env.example .env
```

Nội dung mặc định của `.env`:
```ini
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/semantic_search
EMBEDDING_MODEL_NAME=paraphrase-multilingual-MiniLM-L12-v2
```

### 5. Chuẩn bị dữ liệu và Seed vào Database

```bash
# Bước 1: Tải và xử lý dataset từ Kaggle
python scripts/prepare_data.py

# Bước 2: Tạo bộ câu hỏi đánh giá
python scripts/build_eval_set.py

# Bước 3: Nạp 3,000 documents vào cơ sở dữ liệu
python scripts/seed.py --clear
```

### 6. Khởi động API Server

```bash
uvicorn app.main:app --reload
```

- API Server: [http://localhost:8000](http://localhost:8000)
- Interactive Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

### 7. Minh họa API Search & Trả về Link Stack Overflow gốc

Khi gửi truy vấn tìm kiếm tới `GET /search`:

```bash
curl -X GET "http://localhost:8000/search?q=how%20to%20restrict%20firebase%20api%20key&top_k=2"
```

Mỗi kết quả trả về sẽ có thêm trường **`url`** dẫn thẳng tới câu hỏi gốc trên Stack Overflow (`https://stackoverflow.com/questions/{so_question_id}`). Điều này giúp người dùng dễ dàng truy cập trực tiếp bài viết gốc để:
- Xem toàn bộ ngữ cảnh, hình ảnh minh họa và mã nguồn chi tiết của bài đăng.
- Tham khảo các câu trả lời khác và câu trả lời được chấp thuận (accepted answers) ngoài phần nội dung chunk được trích xuất.

**Ví dụ JSON Response:**
```json
[
  {
    "id": 2,
    "parent_title": "Restricting Firebase API Keys",
    "content": " but it worked for the Android Key and the Server Key , at least as far as I can tell. However, the Browser Key restrictions appear to not work as Firebase is creating a new Browser Key when I redeploy my application. To sum up my question, I can see that Firebase is auto creating API keys for me, but I cannot find any documentation that talks about how these keys are used for the basic features of Firebase that I'm using. I'm also not entirely sure how I can restrict these keys, especially the Browser Key .",
    "similarity": 0.7173,
    "url": "https://stackoverflow.com/questions/51803372"
  },
  {
    "id": 434,
    "parent_title": "com.google.firebase.database.DatabaseException: Calls to setPersistenceEnabled() must be made before any other usage of FirebaseDatabase instance",
    "content": "I am having a problem when I try to setPersistence in fIREBASE,can someone please explain on how to go about it...",
    "similarity": 0.5649,
    "url": "https://stackoverflow.com/questions/37753991"
  }
]
```

---

## 📊 Nguồn Dataset & Tiền xử lý

- **Nguồn gốc**: [60k Stack Overflow Questions with Quality Rate](https://www.kaggle.com/datasets/imoore/60k-stack-overflow-questions-with-quality-rate) trên Kaggle.
- **Giấy phép (License)**: **MIT License**.
- **Quy trình lọc & tiền xử lý**:
  1. **Lọc chất lượng**: Chỉ giữ lại các câu hỏi có nhãn `Y == "HQ"` (High Quality), loại bỏ hoàn toàn các câu hỏi chất lượng thấp hoặc bị đóng (`LQ_EDIT`, `LQ_CLOSE`).
  2. **Bóc tách HTML**: Sử dụng `BeautifulSoup` (`get_text(separator=" ")`) để loại bỏ toàn bộ thẻ HTML trong trường `Body`, chuẩn hoá khoảng trắng.
  3. **Lọc độ dài**: Loại bỏ các bài viết có nội dung sau khi làm sạch ngắn hơn 50 ký tự (`len(content) >= 50`).
  4. **Phân tách tập dữ liệu**:
     - `data/documents_full.json`: Toàn bộ **14,997** câu hỏi HQ đã được làm sạch.
     - `data/documents_dev.json`: Lấy mẫu ngẫu nhiên **3,000** câu hỏi (`random_state=42`) phục vụ phát triển và kiểm thử nhanh.
     - `data/eval_queries.json`: Bộ **30** truy vấn đánh giá chuẩn.

---

## 🔬 Kết quả 3 Thực nghiệm Đánh giá (Experiments)

### 📌 Experiment 1 — So sánh Chiến lược Chunking

Đánh giá chỉ số **Recall@5** trên 30 câu truy vấn chuẩn với 3 cấu hình kích thước chunk và độ chồng lặp (overlap):

| Cấu hình (chunk_size, overlap) | Tổng số Chunks | Số Hit / Tổng query | Recall@5 |
|:------------------------------:|:--------------:|:-------------------:|:--------:|
| **(300, 50)**                  | 4,053          | 29/30               | **0.9667** |
| **(500, 50)**                  | 3,373          | 29/30               | **0.9667** |
| **(500, 100)**                 | 3,400          | 29/30               | **0.9667** |

> **Nhận xét**:
> - Với các văn bản dạng câu hỏi kỹ thuật Stack Overflow (trung bình từ 200–500 tokens), cả 3 cấu hình đều đạt Recall@5 rất cao (**96.67%**).
> - Cấu hình `(500, 50)` tiết kiệm được khoảng **16.7%** dung lượng lưu trữ vector (3,373 so với 4,053 chunks) mà không làm suy giảm độ chính xác tìm kiếm. Cấu hình `(300, 50)` phù hợp khi cần trích dẫn đoạn code ngắn gọn và tập trung.

---

### 📌 Experiment 2 — Keyword Search vs Semantic Search

So sánh giữa tìm kiếm từ khóa bằng **PostgreSQL Full-Text Search (`tsvector` + GIN Index + `ts_rank`)** và tìm kiếm ngữ nghĩa bằng **pgvector (Cosine Similarity)**:

| Phương pháp tìm kiếm | Bộ truy vấn kiểm thử | Số Hit / Tổng query | Recall@5 |
|:--------------------:|:--------------------:|:-------------------:|:--------:|
| **Keyword (tsvector)** | Verbatim Queries (30 câu trùng tiêu đề) | 30/30 | **1.0000** |
| **Semantic (pgvector)** | Verbatim Queries (30 câu trùng tiêu đề) | 29/30 | **0.9667** |
| **Keyword (tsvector)** | Semantic Challenge (5 câu diễn đạt tự nhiên/từ đồng nghĩa) | 0/5 | **0.0000** |
| **Semantic (pgvector)** | Semantic Challenge (5 câu diễn đạt tự nhiên/từ đồng nghĩa) | 4/5 | **0.8000** |

#### Các ví dụ minh họa Semantic Search chiến thắng vượt trội:

1. **Từ đồng nghĩa & diễn đạt khác (Synonyms & Vocabulary Mismatch)**:
   - **Query**: `"verify container cluster manager status from command line"`
   - **Expected Document**: `"How do I check that a docker host is in swarm mode?"`
   - **Semantic Search**: **HIT** (Tìm thấy chính xác ở Top #1 nhờ hiểu ngữ nghĩa giữa *container cluster manager* và *docker swarm mode*).
   - **Keyword Search**: **MISS** (Trả về rỗng `[]` vì trong văn bản không chứa cụm từ *cluster manager*).

2. **Thuật ngữ kỹ thuật tương đương (Technical Synonyms)**:
   - **Query**: `"extract DOM elements tree using bs4 python parser"`
   - **Expected Document**: `"Get all HTML tags with Beautiful Soup"`
   - **Semantic Search**: **HIT** (Tìm thấy chính xác ở Top #1, mô hình hiểu *bs4* là *Beautiful Soup* và *DOM elements* là *HTML tags*).
   - **Keyword Search**: **MISS** (Không tìm thấy do từ khóa không trùng khớp).

3. **Mô tả triệu chứng lỗi (Symptom Description vs Error Name)**:
   - **Query**: `"CUDA deep learning GPU library loading failure in python"`
   - **Expected Document**: `"ImportError: libcudnn when running a TensorFlow program"`
   - **Semantic Search**: **HIT** (Liên kết ngữ nghĩa giữa *CUDA deep learning library failure* và lỗi *libcudnn / TensorFlow*).
   - **Keyword Search**: **MISS** (Không có từ khóa *CUDA* hay *GPU* trong tiêu đề).

> **Nhận xét**: Keyword search hoạt động xuất sắc khi người dùng tìm đúng chính xác từ khóa có trong văn bản, nhưng hoàn toàn bất lực khi người dùng dùng từ đồng nghĩa, mô tả vấn đề hoặc cách diễn đạt khác. Semantic Search giải quyết triệt để vấn đề "từ vựng không khớp" (vocabulary mismatch).

---

### 📌 Experiment 3 — Benchmark Độ trễ Vector Index (Flat Scan vs HNSW)

Đo lường độ trễ trung bình của 30 câu truy vấn (mỗi câu lặp lại 3 lần) trên các mốc quy mô dữ liệu:

| Số lượng Document | Tổng số Chunks | Latency không index - Flat Scan (ms) | Latency có index - HNSW (ms) | Tăng tốc (Speedup) |
|:-----------------:|:--------------:|:-----------------------------------:|:----------------------------:|:------------------:|
| **5,000**         | 6,530          | 18.78 ms                            | **1.31 ms**                  | **14.4x**          |
| **10,000**        | 13,206         | 27.23 ms                            | **2.20 ms**                  | **12.4x**          |
| **14,997 (Full)** | 20,407         | 41.93 ms                            | **1.55 ms**                  | **27.0x**          |

#### 📈 Biểu đồ so sánh độ trễ:

![Latency Chart](data/latency_chart.png)

> **Nhận xét**:
> - **Flat Scan (Quét tuần tự)**: Độ trễ tăng tuyến tính theo số lượng vector ($O(N)$), từ 18.78 ms lên 41.93 ms khi dữ liệu tăng lên 20,000 chunks.
> - **HNSW Index**: Độ trễ duy trì ổn định ở mức cực thấp **~1.3 – 2.2 ms** ($O(\log N)$).
> - Tại quy mô ~20,000 chunks, HNSW giúp truy vấn nhanh hơn tới **27 lần** so với quét tuần tự, bảo đảm khả năng mở rộng (scalability) cho hệ thống trong môi trường production.

---

## 🌐 Hỗ trợ tiếng Việt (Cross-Lingual Search)

### 1. Đổi sang mô hình Embedding đa ngôn ngữ
Ban đầu, hệ thống sử dụng `all-MiniLM-L6-v2` vốn chỉ được huấn luyện trên ngữ liệu tiếng Anh. Do đó, khi người dùng tìm kiếm bằng câu hỏi tiếng Việt, mô hình đơn ngữ không thể ánh xạ ngữ nghĩa giữa tiếng Việt và tài liệu tiếng Anh vào cùng một vùng không gian vector, dẫn tới kết quả tìm kiếm không chính xác.

Hệ thống đã được chuyển đổi sang **`paraphrase-multilingual-MiniLM-L12-v2`**:
- **Hỗ trợ hơn 50 ngôn ngữ**: Huấn luyện đặc thù để đưa các câu mang cùng ngữ nghĩa ở các ngôn ngữ khác nhau về gần nhau trong không gian vector.
- **Kích thước vector không đổi (384 chiều)**: Giữ nguyên cấu trúc schema `VECTOR(384)` và HNSW index trong PostgreSQL, không cần migrate cấu trúc bảng.
- **Dữ liệu được nạp lại (re-seed)**: Toàn bộ vector nhúng của tài liệu tiếng Anh được sinh mới để bảo đảm tính tương thích không gian vector đồng nhất.

### 2. Đánh giá chất lượng Recall@5 (Tiếng Anh vs Tiếng Việt)

Thực hiện kiểm thử trên bộ 30 câu hỏi song ngữ ([data/eval_queries.json](data/eval_queries.json) và [data/eval_queries_vi.json](data/eval_queries_vi.json)):

| Ngôn ngữ query | Tổng số Query | Số Hit (Top-5) | Recall@5 |
|:--------------:|:-------------:|:--------------:|:--------:|
| **Tiếng Anh (English)**    | 30 | 27 | **0.9000** (90.0%) |
| **Tiếng Việt (Vietnamese)** | 30 | 25 | **0.8333** (83.3%) |

### 3. Ví dụ thực tế đánh giá chất lượng

#### ✅ Các trường hợp Tiếng Việt tìm kiếm rất tốt (Hit đúng Top-5):
1. **Dịch tự nhiên kèm thuật ngữ kỹ thuật**:
   - **Query tiếng Việt**: `"Lỗi ImportError libcudnn khi chạy chương trình TensorFlow"`
   - **Expected Document**: `"ImportError: libcudnn when running a TensorFlow program"`
   - **Thực tế**: **HIT** ở vị trí **#2** (Cosine similarity: `0.8172`).
2. **Hành động & thiết bị di động**:
   - **Query tiếng Việt**: `"Cách đổi icon thông báo push notification trong Cordova Android"`
   - **Expected Document**: `"Push notification icon in Cordova Android"`
   - **Thực tế**: **HIT** ở vị trí **#1** (Cosine similarity: `0.6387`).
3. **Lỗi Migration thư viện Android**:
   - **Query tiếng Việt**: `"Lỗi ClassNotFoundException không tìm thấy class android.support.v4.content.FileProvider sau khi migrate sang androidx"`
   - **Expected Document**: `"ClassNotFoundException: Didn't find class "android.support.v4.content.FileProvider" after androidx migration"`
   - **Thực tế**: **HIT** ở vị trí **#1** (Cosine similarity: `0.8046`).

#### ⚠️ Phân tích hạn chế (Các trường hợp Miss):
- **Bản chất bài toán**: Recall tiếng Việt (83.33%) thấp hơn tiếng Anh (90.00%) do tìm kiếm xuyên ngôn ngữ (*cross-lingual search*) luôn khó hơn tìm kiếm cùng ngôn ngữ (*same-language search*).
- **Đặc thù thuật ngữ lập trình**: Nhiều khái niệm kỹ thuật không có từ dịch chuẩn trong tiếng Việt hoặc được pha trộn giữa tiếng Việt và tiếng Anh (vd: `"FlatList"`, `"Mono"`, `"kapt"`, `"lstlisting"`, `"null pointer"`). Khi câu hỏi được thuần Việt hoá (như `"Cách chèn đoạn code trong LaTeX không hiện số dòng bằng môi trường lstlisting"`), sự cách biệt từ vựng với tiêu đề gốc (`"how to do code listing in Latex without line numbers..."`) khiến khoảng cách vector bị kéo dãn và trượt khỏi Top-5.

---

## 🛠 Hướng dẫn chạy lại các thực nghiệm (Reproduce Experiments)

Bạn có thể tự chạy lại từng thực nghiệm độc lập bằng các lệnh sau:

### Re-seed dữ liệu với Model Đa ngôn ngữ (và so_question_id)
```bash
python scripts/reseed.py
```

### Migration thêm cột so_question_id cho bảng hiện có (không cần re-seed)
```bash
python scripts/add_so_question_id_column.py
```

### Tạo bộ dữ liệu đánh giá Tiếng Việt song song
```bash
python scripts/build_eval_set_vi.py
```
*Kết quả lưu tại*: `data/eval_queries_vi.json`

### Chạy Thực nghiệm Cross-Lingual (Tiếng Anh vs Tiếng Việt)
```bash
python scripts/experiment_cross_lingual.py
```
*Kết quả lưu tại*: `data/results_cross_lingual.json`

### Chạy Experiment 1: Đánh giá Chunking
```bash
python scripts/experiment_chunking.py
```
*Kết quả lưu tại*: `data/results_chunking.json`

### Chạy Experiment 2: So sánh Keyword vs Semantic
```bash
python scripts/experiment_keyword_vs_semantic.py
```
*Kết quả lưu tại*: `data/results_keyword_vs_semantic.json`

### Chạy Experiment 3: Benchmark HNSW Index Latency
```bash
python scripts/experiment_index_latency.py
```
*Kết quả lưu tại*: `data/results_index_latency.json` và biểu đồ `data/latency_chart.png`

---

## 🧪 Chạy Kiểm thử tự động (Unit & Integration Tests)

```bash
pytest -v
```

Toàn bộ **11 test cases** (kiểm tra `/health`, validation API, luồng `/documents` & `/search`, logic chunking và overlap token, vector embedding dimension) đều được kiểm thử và bảo đảm vượt qua 100%.
