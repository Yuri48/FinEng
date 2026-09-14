"""REPORT.md -> report.html -> PDF (headless Chrome). Usage: python scripts/build_report.py [out.pdf]"""
import subprocess, sys
from pathlib import Path
import markdown

ROOT = Path(__file__).resolve().parents[1]
CSS = """
body { font-family: Georgia, 'Times New Roman', serif; font-size: 10.5pt; line-height: 1.38; max-width: 7.1in; margin: 0 auto; color: #111; }
h1 { font-size: 17pt; margin-bottom: 2pt; } h2 { font-size: 13pt; margin-top: 16pt; border-bottom: 1px solid #999; padding-bottom: 2pt; }
h3 { font-size: 11pt; margin-top: 12pt; } p.meta { color: #444; margin-top: 0; }
table { border-collapse: collapse; font-size: 8.6pt; margin: 6pt 0 8pt 0; width: 100%; }
th, td { border-bottom: 1px solid #ccc; padding: 2.5pt 4pt; text-align: right; vertical-align: top; }
th { border-bottom: 1.5px solid #333; background: #f3f3f3; } td:first-child, th:first-child { text-align: left; white-space: nowrap; }
img { max-width: 100%; } .caption { font-size: 9pt; color: #333; } blockquote { margin-left: 1em; color: #333; }
code { font-family: Menlo, monospace; font-size: 9pt; } hr { border: 0; border-top: 1px solid #bbb; }
@page { size: Letter; margin: 0.75in 0.7in; }
"""
md = (ROOT / "REPORT.md").read_text(encoding="utf-8")
body = markdown.markdown(md, extensions=["tables", "attr_list", "md_in_html", "sane_lists"])
html = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
(ROOT / "report.html").write_text(html, encoding="utf-8")
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "report.pdf"
chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={out}", f"file://{ROOT / 'report.html'}"], check=True, capture_output=True)
print("wrote", out)
