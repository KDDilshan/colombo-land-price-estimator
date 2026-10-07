"""PDF export for a saved estimate, via fpdf2 (pure Python, no system deps -
installs cleanly on Windows, unlike WeasyPrint which needs GTK)."""

import io

from fpdf import FPDF

import predictor

# Mirrors the site's CSS custom properties (app/static/style.css :root), so the
# report doesn't look like a different product from the page it was exported from.
BRAND_BLUE = (29, 78, 216)   # --accent
INK = (22, 32, 44)           # --ink
MUTED = (93, 107, 124)       # --muted
LINE = (217, 224, 232)       # --line
ROW_ALT = (244, 246, 249)    # --bg


class ReportPDF(FPDF):
    def header(self):
        self.set_fill_color(*BRAND_BLUE)
        self.rect(0, 0, self.w, 22, style="F")
        self.set_xy(12, 6)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 6, "Colombo Land Price Estimator", new_x="LMARGIN", new_y="NEXT")
        self.set_x(12)
        self.set_font("Helvetica", "", 9)
        self.cell(0, 5, "Estimate report", new_x="LMARGIN", new_y="NEXT")
        self.set_y(28)
        self.set_text_color(*INK)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def _kv_table(pdf, rows, title=None):
    if title:
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(*INK)
        pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    label_w = 62
    value_w = pdf.w - pdf.l_margin - pdf.r_margin - label_w
    for i, (label, value) in enumerate(rows):
        fill = i % 2 == 0
        pdf.set_fill_color(*ROW_ALT)
        pdf.set_draw_color(*LINE)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*MUTED)
        pdf.cell(label_w, 8, label, border="B", fill=fill)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*INK)
        pdf.cell(value_w, 8, str(value), border="B", fill=fill, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)


def _flood_row(estimate):
    """Recomputed from the same per-division table predictor.py itself uses -
    the estimate record only stores the price, not every model input."""
    if estimate.known_division and estimate.division in predictor.DIVISION_VALUES.index:
        return predictor.DIVISION_VALUES.loc[estimate.division]
    return predictor.GLOBAL_ROW


def build_pdf(estimate):
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 17)
    pdf.set_text_color(*INK)
    pdf.cell(0, 9, (estimate.division or "Unknown division").title(), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 6, f"Estimate #{estimate.id}  ·  {estimate.created_at:%Y-%m-%d %H:%M} UTC",
              new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # headline price block
    box_w = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.set_fill_color(*ROW_ALT)
    pdf.set_draw_color(*LINE)
    pdf.rect(pdf.l_margin, pdf.get_y(), box_w, 24, style="FD")
    y = pdf.get_y()

    pdf.set_xy(pdf.l_margin + 6, y + 4)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*MUTED)
    pdf.cell(box_w / 2 - 6, 5, "PRICE PER PERCH")
    pdf.set_xy(pdf.l_margin + box_w / 2, y + 4)
    pdf.cell(box_w / 2 - 6, 5, "TOTAL FOR THE PLOT")

    pdf.set_xy(pdf.l_margin + 6, y + 10)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*BRAND_BLUE)
    pdf.cell(box_w / 2 - 6, 8, f"LKR {estimate.price_per_perch:,.0f}")
    pdf.set_xy(pdf.l_margin + box_w / 2, y + 10)
    pdf.cell(box_w / 2 - 6, 8, f"LKR {estimate.total_price:,.0f}")

    pdf.set_y(y + 24 + 8)
    pdf.set_text_color(*INK)

    _kv_table(pdf, [
        ("Land size", f"{estimate.perches:g} perches"),
        ("Land type", estimate.land_type_label or estimate.land_type or "-"),
        ("Basis", "Known division" if estimate.known_division else "Colombo-wide average (unknown division)"),
        ("Confidence", (estimate.confidence or "n/a").title()),
    ], title="Plot details")

    rec = predictor.flood_record(estimate.division)
    if rec is not None:
        def _map(pct):
            if pct is None:
                return "outside the mapped area"
            return f"{pct:.0f}% of the GN inside the flood extent" if pct > 0 else "mapped - not flooded"
        _kv_table(pdf, [
            ("Indication", rec["indication"]),
            ("Survey Dept. flood map, May 2016", _map(rec["flooded_pct_2016"])),
            ("Survey Dept. flood map, May 2018", _map(rec["flooded_pct_2018"])),
            ("Survey Dept. flood map, Nov 2025", _map(rec["flooded_pct_2025"])),
            ("DMC records naming this GN", ", ".join(map(str, rec["dmc_years"])) or "none"),
        ], title="Past flooding - official records")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*MUTED)
        pdf.multi_cell(0, 5, "Descriptive only, from Survey Department of Sri Lanka flood maps "
                              "and the DMC DesInventar database. Not a prediction and not used in "
                              "the price; a GN with no record may still flood.")
        pdf.ln(4)

    row = _flood_row(estimate)
    _kv_table(pdf, [
        ("Surface water occurrence", f"{row.get('water_occurrence_pct', 0):.2f}%"),
        ("Height above nearest drainage", f"{row.get('hand_m', 0):.1f} m"),
        ("Distance to nearest stream", f"{row.get('dist_to_stream_km', 0):.2f} km"),
        ("Vegetation (NDVI)", f"{row.get('ndvi', 0):.2f}"),
    ], title="Environmental and spatial profile")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(0, 5, "Values the price model uses for this division. They are associations "
                          "with asking prices, not a flood forecast or a causal effect.")
    pdf.ln(4)

    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(
        0, 5,
        f"Estimate only, not a valuation. Model accuracy: median error {predictor.MEDIAN_APE}%, "
        f"mean error {predictor.MEAN_APE}%, R2 {predictor.R2_KNOWN}% on held-out listings "
        f"from the modelled divisions.",
    )

    return io.BytesIO(pdf.output())
