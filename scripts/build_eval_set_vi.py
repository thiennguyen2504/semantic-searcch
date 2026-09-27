"""Build parallel Vietnamese evaluation queries from data/eval_queries.json."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EN_EVAL_FILE = PROJECT_ROOT / "data" / "eval_queries.json"
VI_EVAL_FILE = PROJECT_ROOT / "data" / "eval_queries_vi.json"

# Manual human translation mapping for all 30 queries
# Tailored for natural Vietnamese technical question phrasing
MANUAL_TRANSLATIONS = {
    "ImportError: libcudnn when running a TensorFlow program": (
        "Lỗi ImportError libcudnn khi chạy chương trình TensorFlow"
    ),
    "Push notification icon in Cordova Android": (
        "Cách đổi icon thông báo push notification trong Cordova Android"
    ),
    "Get current filename in Babel Plugin?": (
        "Lấy tên file hiện tại trong Babel Plugin như thế nào?"
    ),
    "RxJava: How to convert List of objects to List of another objects": (
        "RxJava: Cách chuyển đổi List object này sang List object khác"
    ),
    "ClassNotFoundException: Didn't find class \"android.support.v4.content.FileProvider\" after androidx migration": (
        "Lỗi ClassNotFoundException không tìm thấy class android.support.v4.content.FileProvider sau khi migrate sang androidx"
    ),
    "Can you declare a object literal type that allows unknown properties in typescript?": (
        "Có thể khai báo kiểu object literal cho phép thuộc tính không xác định trong typescript không?"
    ),
    "How do I check that a docker host is in swarm mode?": (
        "Làm thế nào để kiểm tra docker host đang ở chế độ swarm mode?"
    ),
    "Can I detect element visibility using only CSS?": (
        "Có thể nhận biết phần tử đang hiển thị hay ẩn chỉ bằng CSS không?"
    ),
    "SAML Signing Certificate - Which SSL Certificate Type?": (
        "Chứng chỉ SAML Signing Certificate nên dùng loại SSL Certificate nào?"
    ),
    "How to Disable Start Page After Solution Close in Visual Studio 2017": (
        "Cách tắt Start Page sau khi đóng Solution trong Visual Studio 2017"
    ),
    "Creating a local notification in response to a push notification (from firebase) in cordova/ionic": (
        "Tạo local notification khi nhận push notification từ firebase trong cordova/ionic"
    ),
    "Upgrade Angular version (now: 2.4.3 or 4.0.0-beta.3) following the best practice?": (
        "Cách nâng cấp phiên bản Angular đúng chuẩn best practice?"
    ),
    "significance of \"trainable\" and \"training\" flag in tf.layers.batch_normalization": (
        "Ý nghĩa của flag trainable và training trong tf.layers.batch_normalization"
    ),
    "Get all HTML tags with Beautiful Soup": (
        "Lấy tất cả các thẻ HTML bằng Beautiful Soup trong python"
    ),
    "iOS 10: How to debug a Today Widget - \"Unable to load\" message": (
        "iOS 10: Cách debug Today Widget khi gặp thông báo Unable to load"
    ),
    "Ionic with Android Emulator: Automatically send location?": (
        "Ionic chạy trên Android Emulator: Làm sao tự động gửi tọa độ vị trí location?"
    ),
    "Which way to name a function in Go, CamelCase or Semi-CamelCase?": (
        "Quy tắc đặt tên hàm trong Go nên dùng CamelCase hay Semi-CamelCase?"
    ),
    "Firebase notifications not working in iOS 11": (
        "Thông báo Firebase notification không hoạt động trên iOS 11"
    ),
    "How to keep scroll position using flatlist when navigating back in react native ?": (
        "Cách giữ nguyên vị trí cuộn scroll position của FlatList khi back lại trang trong React Native?"
    ),
    "ReportLab: working with Chinese/Unicode characters": (
        "ReportLab: cách hiển thị ký tự tiếng Trung hoặc Unicode trong PDF"
    ),
    "How do I flatten a tensor in pytorch?": (
        "Làm thế nào để làm phẳng flatten một tensor trong PyTorch?"
    ),
    "Python - Most elegant way to extract a substring, being given left and right borders": (
        "Python: Cách tốt nhất để trích xuất chuỗi con substring nằm giữa hai mốc ký tự trái phải"
    ),
    "Type '() => void' is not assignable to type '() => {}'": (
        "Lỗi Type '() => void' is not assignable to type '() => {}' trong TypeScript"
    ),
    "Duplicate GlobalKey detected in widget tree": (
        "Lỗi phát hiện trùng lặp Duplicate GlobalKey trong widget tree của Flutter"
    ),
    "Python JSON dummy data generation from JSON schema": (
        "Tạo dữ liệu giả dummy data dạng JSON từ JSON schema trong Python"
    ),
    "Running a Jupyter notebook from another notebook": (
        "Cách chạy một Jupyter notebook từ một notebook khác"
    ),
    "How do I use Reactor's StepVerifier to verify a Mono is empty?": (
        "Cách dùng StepVerifier của Reactor để kiểm tra một Mono rỗng?"
    ),
    "how to do code listing in Latex without line numbers (when using lstlisting environment)": (
        "Cách chèn đoạn code trong LaTeX không hiện số dòng bằng môi trường lstlisting"
    ),
    "Sandbox subdomains are for test purposes only. Please add your own domain or add the address to authoriz": (
        "Thông báo lỗi Sandbox subdomains are for test purposes only trong Mailgun"
    ),
    "Databinding annotation processor kapt warning": (
        "Cảnh báo Databinding annotation processor khi build với kapt trong Android Kotlin"
    ),
}


def build_vi_eval_set():
    if not EN_EVAL_FILE.exists():
        print(f"Error: {EN_EVAL_FILE} not found!")
        sys.exit(1)

    with open(EN_EVAL_FILE, "r", encoding="utf-8") as f:
        en_queries = json.load(f)

    vi_queries = []
    missing_translations = []

    for item in en_queries:
        en_q = item["query"]
        parent_title = item["expected_parent_title"]

        if en_q in MANUAL_TRANSLATIONS:
            vi_q = MANUAL_TRANSLATIONS[en_q]
        else:
            vi_q = f"TODO: dịch câu này [{en_q}]"
            missing_translations.append(en_q)

        vi_queries.append({
            "query": vi_q,
            "expected_parent_title": parent_title,
        })

    VI_EVAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(VI_EVAL_FILE, "w", encoding="utf-8") as f:
        json.dump(vi_queries, f, ensure_ascii=False, indent=2)

    print(f"Created {VI_EVAL_FILE} with {len(vi_queries)} queries.")
    if missing_translations:
        print(f"WARNING: {len(missing_translations)} queries have TODO placeholders.")
    else:
        print("All 30 queries translated successfully with natural technical Vietnamese.")


if __name__ == "__main__":
    build_vi_eval_set()
