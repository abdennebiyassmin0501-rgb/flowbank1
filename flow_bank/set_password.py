"""
set_password.py — إنشاء أو تغيير حساب المستخدم الوحيد.

    python set_password.py

يكتب الملف .streamlit/secrets.toml (للتشغيل على جهازك) ويطبع النص نفسه
لتلصقيه في خانة Secrets على Streamlit Community Cloud.
الملف secrets.toml مستثنى من GitHub في .gitignore، فلا يُرفع أبدًا.
"""
import getpass
import sys
from pathlib import Path

from auth import make_record, to_toml

SECRETS = Path(__file__).parent / ".streamlit" / "secrets.toml"


def main():
    if len(sys.argv) == 3:
        username, password = sys.argv[1], sys.argv[2]
    else:
        username = input("اسم المستخدم: ").strip()
        password = getpass.getpass("كلمة المرور (لن تظهر أثناء الكتابة): ")
        if password != getpass.getpass("أعيدي كتابة كلمة المرور: "):
            print("كلمتا المرور غير متطابقتين. لم يتغير شيء.")
            return
    if not username:
        print("اسم المستخدم مطلوب.")
        return
    if len(password) < 8:
        print("كلمة المرور يجب أن تكون 8 أحرف على الأقل.")
        return
    text = to_toml(make_record(username, password))
    SECRETS.parent.mkdir(exist_ok=True)
    SECRETS.write_text(text, encoding="utf-8")
    print(f"تم الحفظ في {SECRETS}.\n")
    print("للنشر على Streamlit Cloud، انسخي هذه الأسطر والصقيها في Settings ← Secrets:\n")
    print(text)


if __name__ == "__main__":
    main()
