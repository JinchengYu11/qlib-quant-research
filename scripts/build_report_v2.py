"""生成 PROJECT_REPORT_v2.pdf: PROJECT_REPORT.md → HTML → Playwright PDF.

跟 build_report.py 不同, v2 走 markdown 渲染路, 直接读源 .md 文件,
不再 hardcode HTML. 适合 v2 内容多 + 频繁更新.
"""
import re
from pathlib import Path
import markdown
from playwright.sync_api import sync_playwright

PROJ = Path("/Users/cedricyu/qlib量化研究")
SRC = PROJ / "PROJECT_REPORT.md"
OUT_HTML = PROJ / "results" / "report_v2.html"
OUT_PDF = PROJ / "results" / "PROJECT_REPORT_v2.pdf"


CSS = """
@page {
  size: A4;
  margin: 18mm 16mm 20mm 16mm;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body {
  font-family: 'PingFang HK', 'Hiragino Sans GB', 'Heiti TC', 'Arial', sans-serif;
  color: #2d3748;
  line-height: 1.65;
  font-size: 10pt;
}
body { padding: 0 4mm; }

h1, h2, h3, h4 {
  color: #1a365d;
  line-height: 1.3;
  font-weight: 600;
  margin-top: 1.8em;
  margin-bottom: 0.6em;
  page-break-after: avoid;
}
h1 {
  font-size: 22pt;
  border-bottom: 3px solid #c89b3c;
  padding-bottom: 4mm;
  margin-top: 0;
}
h1:not(:first-child) {
  page-break-before: always;
  margin-top: 0;
}
h2 {
  font-size: 14pt;
  border-left: 4px solid #c89b3c;
  padding-left: 8px;
  page-break-before: auto;
}
h3 { font-size: 12pt; color: #2c5282; }
h4 { font-size: 11pt; color: #4a5568; }

p { margin: 0.6em 0; }
em { color: #c89b3c; font-style: normal; font-weight: 600; }
strong { color: #1a365d; font-weight: 700; }

ul, ol { margin: 0.5em 0 0.8em 1.6em; }
li { margin: 0.2em 0; }

blockquote {
  border-left: 4px solid #c89b3c;
  background: #fefcf6;
  padding: 8px 14px;
  margin: 0.8em 0;
  color: #4a5568;
  font-size: 9.5pt;
  font-style: italic;
}

code {
  font-family: 'Menlo', 'Monaco', monospace;
  background: #f0f4f8;
  padding: 1px 4px;
  border-radius: 3px;
  font-size: 9pt;
  color: #c53030;
}

pre {
  background: #1a202c;
  color: #e2e8f0;
  padding: 10px 14px;
  border-radius: 4px;
  font-size: 8.5pt;
  line-height: 1.5;
  overflow-x: auto;
  margin: 0.8em 0;
  page-break-inside: avoid;
}
pre code { background: none; color: inherit; padding: 0; }

table {
  width: 100%;
  border-collapse: collapse;
  margin: 0.8em 0;
  font-size: 9pt;
  page-break-inside: avoid;
}
th {
  background: #1a365d;
  color: white;
  padding: 6px 8px;
  text-align: left;
  font-weight: 600;
}
td {
  padding: 5px 8px;
  border-bottom: 1px solid #e2e8f0;
}
tr:nth-child(even) td { background: #f7fafc; }

hr {
  border: none;
  border-top: 1px solid #cbd5e0;
  margin: 1.5em 0;
}

img {
  max-width: 100%;
  display: block;
  margin: 0.8em auto;
  page-break-inside: avoid;
}

.toc {
  background: #f7fafc;
  border: 1px solid #cbd5e0;
  border-radius: 5px;
  padding: 12px 18px;
  margin: 1em 0;
}
.toc h3 { margin-top: 0; }

/* 高亮特殊段落: 反证、突破 */
strong:has(+ em), em:has(+ strong) { background: #fef3c7; padding: 1px 3px; border-radius: 2px; }
"""


def md_to_html(md_text):
    md = markdown.Markdown(extensions=['extra', 'tables', 'fenced_code', 'sane_lists'])
    body = md.convert(md_text)

    # Fix image paths: relative → absolute file:// URLs
    def fix_img(match):
        path = match.group(1)
        if path.startswith('http'):
            return match.group(0)
        abs_path = (PROJ / path).resolve()
        return f'src="file://{abs_path}"'

    body = re.sub(r'src="([^"]+)"', fix_img, body)

    return body


def build_full_html(body):
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>A 股量化策略研究 v2</title>
<style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>"""


def main():
    print(f"[1] 读 {SRC}", flush=True)
    md_text = SRC.read_text(encoding="utf-8")
    print(f"    {len(md_text)} 字符, {md_text.count(chr(10))} 行", flush=True)

    print(f"[2] Markdown → HTML", flush=True)
    body = md_to_html(md_text)
    html = build_full_html(body)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"    HTML → {OUT_HTML} ({len(html)/1024:.0f} KB)", flush=True)

    print(f"[3] Playwright → PDF", flush=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(f"file://{OUT_HTML.resolve()}", wait_until="networkidle")
        page.pdf(
            path=str(OUT_PDF),
            format="A4",
            print_background=True,
            margin={"top": "18mm", "right": "16mm", "bottom": "20mm", "left": "16mm"},
        )
        browser.close()
    size_kb = OUT_PDF.stat().st_size / 1024
    print(f"    PDF → {OUT_PDF} ({size_kb:.0f} KB)", flush=True)
    print("[DONE]", flush=True)


if __name__ == "__main__":
    main()
