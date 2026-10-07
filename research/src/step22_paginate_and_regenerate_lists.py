"""
step22_paginate_and_regenerate_lists.py
---------------------------------------
Give THESIS_FINAL.docx real page numbers, then rebuild its Table of Contents,
List of Figures, List of Tables and List of Listings against them.

The document had no page numbering of any kind: no footer, no PAGE field, one
section. Its contents lists were plain typed text carrying page numbers that
matched no pagination the document could produce, which is why the audit found
them stale. Fixing the numbers without fixing that would just produce a
different set of numbers pointing at nothing.

So this runs in three passes.

  1  Structure. Insert a section break before Chapter 1 so the front matter can
     be numbered in lower-case roman and the body in arabic restarting at 1,
     which is the convention the existing lists already assume. Add a centred
     PAGE field to each section's footer.

  2  Measure. Drive Word to open the document and report, for every heading and
     every caption, the page it actually falls on. Word is the only thing that
     paginates a .docx authoritatively, so the numbers come from it rather than
     from an estimate.

  3  Rewrite. Replace the page number on every line of the four lists with the
     measured one, matching each entry to its heading or caption by text. An
     entry that cannot be matched is reported rather than guessed at.

Word is then used once more to export the finished PDF.
"""

from __future__ import annotations

import copy
import os
import re
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TF = os.path.join(ROOT, "thesis_final")
DOCX = os.path.join(TF, "THESIS_FINAL.docx")
PDF = os.path.join(TF, "THESIS_FINAL.pdf")

BODY_START = "Chapter 1: Introduction"


# ------------------------------------------------------------------ pass 1

def _page_field_paragraph(footer):
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    for r in list(p.runs):
        r._r.getparent().remove(r._r)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"
    for el, attrs, text in (("w:fldChar", {"w:fldCharType": "begin"}, None),
                            ("w:instrText", {XML_SPACE: "preserve"}, " PAGE "),
                            ("w:fldChar", {"w:fldCharType": "separate"}, None),
                            ("w:t", {}, "1"),
                            ("w:fldChar", {"w:fldCharType": "end"}, None)):
        e = OxmlElement(el)
        for k, v in attrs.items():
            e.set(k if k.startswith("{") else qn(k), v)
        if text is not None:
            e.text = text
        run._r.append(e)


def _set_pgnum(sectPr, fmt, start):
    for old in sectPr.findall(qn("w:pgNumType")):
        sectPr.remove(old)
    e = OxmlElement("w:pgNumType")
    e.set(qn("w:fmt"), fmt)
    e.set(qn("w:start"), str(start))
    # pgNumType must precede pgSz/pgMar-adjacent elements; appending is accepted
    sectPr.append(e)


def add_pagination(path):
    d = Document(path)
    if len(d.sections) == 2:
        print("  pass 1: already paginated, left alone")
        return
    body = d.element.body
    final_sectPr = body.find(qn("w:sectPr"))
    if final_sectPr is None:
        raise SystemExit("no body sectPr")

    target = None
    for p in d.paragraphs:
        if p.text.strip() == BODY_START:
            target = p
            break
    if target is None:
        raise SystemExit(f"heading not found: {BODY_START}")

    # a paragraph carrying a sectPr ends the section it sits in, so the break
    # paragraph goes immediately before the first body heading
    brk = OxmlElement("w:p")
    pPr = OxmlElement("w:pPr")
    front_sectPr = copy.deepcopy(final_sectPr)
    _set_pgnum(front_sectPr, "lowerRoman", 1)
    pPr.append(front_sectPr)
    brk.append(pPr)
    target._p.addprevious(brk)

    _set_pgnum(final_sectPr, "decimal", 1)
    d.save(path)

    d = Document(path)
    if len(d.sections) != 2:
        raise SystemExit(f"expected 2 sections, got {len(d.sections)}")
    for s in d.sections:
        s.footer.is_linked_to_previous = False
        _page_field_paragraph(s.footer)
    d.save(path)
    print(f"  pass 1: 2 sections, roman front matter, arabic body, PAGE footers")


# ------------------------------------------------------------------ pass 2

def measure(path):
    import win32com.client as win32
    word = win32.gencache.EnsureDispatch("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    try:
        doc = word.Documents.Open(os.path.abspath(path), ReadOnly=False,
                                  AddToRecentFiles=False)
        doc.Repaginate()
        wdActiveEndPageNumber = 3
        wdActiveEndAdjustedPageNumber = 1
        out = []
        for i in range(1, doc.Paragraphs.Count + 1):
            p = doc.Paragraphs(i)
            txt = p.Range.Text.replace("\r", "").replace("\x07", "").strip()
            if not txt:
                continue
            style = str(p.Style.NameLocal)
            if not (style.lower().startswith("heading")
                    or txt.startswith(("Fig. ", "Figure ", "Table ", "Listing "))):
                continue
            # a line in the Table of Contents or one of the lists ends with a
            # tab and its own page number; those are entries, not anchors, and
            # letting one in makes every figure resolve to the List of Figures
            flat = p.Range.Text.replace(chr(13), '').replace(chr(7), '')
            if re.search(chr(9) + r'\s*([ivxlcdm]+|\d+)\s*$', flat, flags=re.I):
                continue
            page = int(p.Range.Information(wdActiveEndAdjustedPageNumber))
            out.append((txt, style, page))
        total = int(doc.ComputeStatistics(2))  # wdStatisticPages
        doc.Close(SaveChanges=-1)
        print(f"  pass 2: Word paginated {total} pages, {len(out)} anchors measured")
        return out
    finally:
        word.Quit()


def export_pdf(path, pdf):
    import win32com.client as win32
    word = win32.gencache.EnsureDispatch("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    try:
        doc = word.Documents.Open(os.path.abspath(path), ReadOnly=True,
                                  AddToRecentFiles=False)
        doc.SaveAs(os.path.abspath(pdf), FileFormat=17)
        doc.Close(SaveChanges=0)
    finally:
        word.Quit()
    print(f"  exported {pdf}  ({os.path.getsize(pdf):,} bytes)")


# ------------------------------------------------------------------ pass 3

def norm(s):
    s = s.replace("’", "'").replace("—", "-").replace("–", "-")
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


ROMAN = ["", "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi",
         "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx"]


def rewrite_lists(path, anchors, front_pages):
    d = Document(path)

    by_text = {}
    for txt, style, page in anchors:
        by_text.setdefault(norm(txt), page)

    # caption anchors keyed by their label, e.g. "fig. 6.2" / "table 5.4"
    by_label = {}
    for txt, style, page in anchors:
        m = re.match(r"(Fig\.|Figure|Table|Listing)\s+([0-9]+\.[0-9]+|[A-Z]\.[0-9]+)",
                     txt)
        if m:
            key = norm(f"{m.group(1)} {m.group(2)}").replace("figure", "fig.")
            by_label.setdefault(key, page)

    changed, unmatched = 0, []
    in_list = False
    for p in d.paragraphs:
        t = p.text.strip()
        if not t:
            continue
        if t in ("Table of Contents", "List of Figures", "List of Tables",
                 "List of Listings"):
            in_list = True
            continue
        if t == "List of Abbreviations":
            in_list = False
            continue
        if not in_list:
            continue

        m = re.match(r"^(.*?)\t([ivxlcdm]+|\d+)\s*$", t, flags=re.I)
        if not m:
            continue
        label = m.group(1).strip()
        page = None

        lm = re.match(r"(Fig\.|Figure|Table|Listing)\s+([0-9]+\.[0-9]+|[A-Z]\.[0-9]+)",
                      label)
        if lm:
            page = by_label.get(norm(f"{lm.group(1)} {lm.group(2)}")
                                .replace("figure", "fig."))
        if page is None:
            page = by_text.get(norm(label))
        if page is None:
            # TOC entries carry a section number the heading also carries
            for k, v in by_text.items():
                if k.startswith(norm(label)) or norm(label).startswith(k):
                    page = v
                    break
        if page is None:
            unmatched.append(label)
            continue

        if label in front_pages:
            new = front_pages[label]
        elif page <= 0:
            new = m.group(2)
        else:
            new = str(page)

        if new != m.group(2):
            _replace_tail(p, m.group(2), new)
            changed += 1

    d.save(path)
    print(f"  pass 3: {changed} page numbers rewritten, {len(unmatched)} unmatched")
    for u in unmatched:
        print(f"       unmatched: {u[:74]}")
    return unmatched


def _replace_tail(p, old, new):
    runs = p.runs
    for r in reversed(runs):
        if r.text.endswith(old):
            r.text = r.text[: -len(old)] + new
            return
    # the number may sit in its own run split across characters
    joined = "".join(r.text for r in runs)
    if joined.endswith(old):
        keep = len(joined) - len(old)
        acc = 0
        for r in runs:
            if acc + len(r.text) <= keep:
                acc += len(r.text)
                continue
            cut = max(0, keep - acc)
            r.text = r.text[:cut]
            acc = keep
        runs[-1].text += new


def main():
    if not os.path.exists(DOCX):
        raise SystemExit(f"missing {DOCX}")
    add_pagination(DOCX)
    anchors = measure(DOCX)

    # front-matter entries keep roman numerals; measure them from section 1
    front = {}
    for txt, style, page in anchors:
        if txt in ("Declaration", "Dedication", "Acknowledgements", "Abstract",
                   "Table of Contents", "List of Figures", "List of Tables",
                   "List of Listings", "List of Abbreviations"):
            front[txt] = ROMAN[page] if page < len(ROMAN) else str(page)

    rewrite_lists(DOCX, anchors, front)
    export_pdf(DOCX, PDF)


if __name__ == "__main__":
    main()
