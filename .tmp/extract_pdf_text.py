from pypdf import PdfReader
from pathlib import Path

pdf_path = Path(r"c:\Users\SebastianRochaext\OneDrive - Prompt Soluciones Integradas, S. de R.L. de C.V\Desktop\pairs_trading_novel\Pairs Trading Using a Novel Graphical Matching Approach.pdf")
out_path = Path(r"c:\Users\SebastianRochaext\OneDrive - Prompt Soluciones Integradas, S. de R.L. de C.V\Desktop\pairs_trading_novel\.tmp\pairs_trading_novel_graphical_matching_extracted.txt")

reader = PdfReader(str(pdf_path))
parts = []
for i, page in enumerate(reader.pages, start=1):
    txt = page.extract_text() or ""
    parts.append(f"\n\n===== PAGE {i} =====\n\n")
    parts.append(txt)

out_path.write_text("".join(parts), encoding="utf-8")
print(f"pages={len(reader.pages)}")
print(f"out={out_path}")
