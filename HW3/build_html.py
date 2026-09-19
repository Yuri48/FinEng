"""Render REPORT.md (narrative + generated tables) to a single self-contained HTML file with embedded figures."""
import re, base64, html, os
def md_to_html(md):
    out = []; lines = md.split("\n"); i = 0; in_list = False
    def inline(t):
        t = html.escape(t, quote=False)
        t = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", lambda m: f'<img alt="{m.group(1)}" src="{embed(m.group(2))}">', t)
        t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t); t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t); t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
        return t
    def embed(p):
        if os.path.exists(p):
            return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()
        return p
    while i < len(lines):
        l = lines[i]
        if l.startswith("|") and i + 1 < len(lines) and re.match(r"^\|\s*-", lines[i + 1]):
            hdr = [c.strip() for c in l.strip("|").split("|")]; i += 2; rows = []
            while i < len(lines) and lines[i].startswith("|"): rows.append([c.strip() for c in lines[i].strip("|").split("|")]); i += 1
            out.append("<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in hdr) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table>"); continue
        if l.lstrip().startswith("<"):                      # raw HTML block (e.g. the title block): pass through untouched
            blk = []
            while i < len(lines) and lines[i].strip(): blk.append(lines[i]); i += 1
            out.append("\n".join(blk)); continue
        m = re.match(r"^(#{1,4})\s+(.*)", l)
        if m: out.append(f"<h{len(m.group(1))}>{inline(m.group(2))}</h{len(m.group(1))}>"); i += 1; continue
        if re.match(r"^\s*[-*]\s+", l):
            if not in_list: out.append("<ul>"); in_list = True
            item = re.sub(r"^\s*[-*]\s+", "", l); out.append(f"<li>{inline(item)}</li>"); i += 1
            if i >= len(lines) or not re.match(r"^\s*[-*]\s+", lines[i]): out.append("</ul>"); in_list = False
            continue
        if re.match(r"^\s*\d+\.\s+", l):
            items = []
            while i < len(lines) and re.match(r"^\s*\d+\.\s+", lines[i]): items.append(re.sub(r"^\s*\d+\.\s+", "", lines[i])); i += 1
            out.append("<ol>" + "".join(f"<li>{inline(x)}</li>" for x in items) + "</ol>"); continue
        if l.strip() == "": i += 1; continue
        para = [l]; i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("|", "#", "- ", "* ", "!")) and not re.match(r"^\s*\d+\.\s+", lines[i]): para.append(lines[i]); i += 1
        out.append(f"<p>{inline(' '.join(para))}</p>")
    return "\n".join(out)
md = open("REPORT.md").read()
css = """body{font-family:Georgia,serif;max-width:1100px;margin:30px auto;padding:0 20px;line-height:1.45;color:#222}h1{font-size:1.7em}h2{border-bottom:1px solid #ccc;margin-top:1.6em}
table{border-collapse:collapse;font-size:0.82em;margin:10px 0;font-family:Helvetica,Arial,sans-serif}th,td{border:1px solid #ddd;padding:3px 7px;text-align:left;vertical-align:top}th{background:#f2f2f2}
img{max-width:100%;border:1px solid #eee;margin:8px 0}code{background:#f5f5f5;padding:1px 3px}p{text-align:justify}p.meta{text-align:left;line-height:1.7}
@page{size:A4;margin:14mm}@media print{body{max-width:none;margin:0;font-size:11pt}table{width:100%;font-size:7.2pt}td,th{word-wrap:break-word;overflow-wrap:anywhere;padding:2px 4px}h2{page-break-after:avoid}img{page-break-inside:avoid}tr{page-break-inside:avoid}}"""
open("REPORT.html", "w").write(f"<!doctype html><html><head><meta charset='utf-8'><title>Iran War Risk 2026</title><style>{css}</style></head><body>{md_to_html(md)}</body></html>")
print("wrote REPORT.html", os.path.getsize("REPORT.html") // 1024, "KB")
