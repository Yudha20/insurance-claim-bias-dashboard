"""Loading skeletons and page-motion assets.

This module deliberately imports nothing heavy. app.py sends its output before
importing pandas and scikit-learn, so the skeleton below appears instead of
Streamlit's generic placeholder, which shows after 500 ms with nothing to draw.

Each skeleton mirrors the real page, using heights measured from the rendered
dashboard. If a page's layout changes, update its entry in PAGES.
"""
import re
from pathlib import Path

HERE = Path(__file__).parent


def slug(view):
    """Match the suffix Streamlit gives a nav button's st-key class."""
    return view.replace(" ", "-")


def _bone(width="100%", height=12, radius=6, style=""):
    w = f"{width}px" if isinstance(width, int) else width
    return f'<div class="sk-bone" style="width:{w};height:{height}px;border-radius:{radius}px;{style}"></div>'


def _line(width, height=20, bone=11):
    return f'<div class="sk-line" style="height:{height}px">{_bone(width, bone)}</div>'


def _stack(*children, gap=16):
    return f'<div class="sk-stack" style="gap:{gap}px">{"".join(children)}</div>'


def _cols(*columns, template=None):
    grid = template or " ".join(["minmax(0,1fr)"] * len(columns))
    return f'<div class="sk-cols" style="grid-template-columns:{grid}">{"".join(columns)}</div>'


def h1(width):
    return f'<div class="sk-h" style="height:40px">{_bone(width, 24, 8)}</div>'


def sub(width):
    return f'<div class="sk-h" style="height:34px">{_bone(width, 18, 7)}</div>'


def intro(*widths):
    return f'<div class="sk-intro">{"".join(_line(w, 22, 12) for w in widths)}</div>'


# Line counts per row of an explanation. Each row is lines x 20px, with 8px between rows.
def explain(*rows):
    fill = ["94%", "88%", "71%", "97%", "63%"]
    out, n = [], 0
    for lines in rows:
        text = []
        for _ in range(lines):
            text.append(_line(fill[n % len(fill)]))
            n += 1
        out.append(f'<div class="sk-explain-label">{_bone(72, 20, 4)}</div><div>{"".join(text)}</div>')
    return f'<div class="sk-explain">{"".join(out)}</div>'


def control(height=68, label=96):
    return f'<div class="sk-control" style="height:{height}px">{_bone(label, 11)}{_bone("100%", 40, 12, "margin-top:auto")}</div>'


def caption(width=120, height=19):
    return _line(width, height, 10)


def button(width, height=40):
    return _bone(width, height, 12)


def bar(height=40):
    return _bone("100%", height, 12)


def table(height, rows, cols, row=35, head=35, widths=None):
    """widths: optional column widths in px, as the real grid sizes them to their content."""
    def sized(i):
        return f"flex:0 0 {widths[i]}px" if widths else "flex:1"
    def cells(w):
        return "".join(f'<div class="sk-cell" style="{sized(i)}">{_bone(w[i % len(w)], 10)}</div>' for i in range(cols))
    top = f'<div class="sk-row sk-head" style="height:{head}px">{cells(["52%", "38%", "46%"])}</div>'
    body = "".join(f'<div class="sk-row" style="height:{row}px">{cells(["72%", "54%", "63%", "44%"] if i % 2 else ["58%", "66%", "48%", "70%"])}</div>'
                   for i in range(rows))
    fit = "width:fit-content;max-width:100%;" if widths else ""
    return f'<div class="sk-table" style="{fit}height:{height}px">{top}{body}</div>'


def hbars(height, count):
    widths = [92, 84, 77, 70, 66, 61, 57, 52, 48, 45, 41, 38, 34, 31, 28, 25, 22, 19, 16]
    rows = "".join(f'<div class="sk-hbar">{_bone("100%", 10)}<div>{_bone(f"{widths[i % len(widths)]}%", "100%", 4)}</div></div>'.replace("height:100%px", "height:100%")
                   for i in range(count))
    return f'<div class="sk-chart sk-hbars" style="height:{height}px">{rows}</div>'


def vbars(height, count):
    heights = [46, 72, 58, 84, 39, 66, 52]
    cols = "".join(f'<div>{_bone("100%", 0, 4, f"height:{heights[i % len(heights)]}%")}</div>' for i in range(count))
    return f'<div class="sk-chart sk-vbars" style="height:{height}px">{cols}</div>'


def curves(height):
    arcs = "".join(f'<i style="inset:{i * 7}% 0 0 {i * 2}%"></i>' for i in range(4))
    return f'<div class="sk-chart sk-curves" style="height:{height}px">{arcs}</div>'


def matrix(height=220):
    return f'<div class="sk-chart sk-matrix" style="height:{height}px">{"".join(_bone("100%", 0, 6, "height:100%") for _ in range(4))}</div>'


def _kpis():
    card = ('<div class="kpi sk-kpi">' + _bone(96, 12) + _bone(112, 32, 8, "margin-top:20px")
            + '<div class="sk-kpi-note">' + _bone(92, 10) + _bone(64, 20, 4) + '</div></div>')
    return f'<div class="sk-kpis">{card * 4}</div>'


def _header():
    title = _stack(_bone(232, 18, 7), _bone(372, 11), gap=12)
    actions = _bone(149, 36, 18) + _bone(134, 36, 18) + _bone(141, 36, 18)
    return f'<div class="sk-header"><div>{title}</div><div class="sk-actions">{actions}</div></div>'


def _sidebar(views):
    widths = [52, 68, 78, 112, 52, 72, 60]
    rows = "".join(f'<div class="sk-nav-row">{_bone(20, 20, 6)}{_bone(widths[i % len(widths)], 12)}</div>' for i in range(len(views)))
    return (f'<div class="sk-side"><div class="sk-brand">{_bone(20, 20, 6)}{_bone(84, 13)}</div>'
            f'<div class="sk-nav">{rows}</div></div>')


FOOT = caption(560)
PAGES = {
    "Claims": [
        intro("96%", "58%"),
        '<div class="sk-card">' + _stack(
            sub(150), explain(2, 1),
            _cols(control(label=92), control(label=56), control(label=40), template="minmax(0,2.04fr) minmax(0,1fr) minmax(0,1fr)"),
            caption(116), table(560, 11, 6, row=48, head=36, widths=[165, 178, 218, 125, 169, 232]), button(158), bar(),
        ) + "</div>",
        FOOT,
    ],
    "Overview": [
        h1("48%"), intro("94%", "62%"), control(label=220), explain(2),
        _cols(_stack(sub("58%"), table(150, 3, 3)), _stack(sub("62%"), table(115, 2, 3))),
        sub("24%"), explain(2, 1), hbars(130, 2),
        sub("20%"), explain(2, 1, 1), table(408, 11, 5), FOOT,
    ],
    "Disparities": [
        h1("46%"), intro("92%", "56%"), bar(71),
        sub("34%"), table(290, 7, 7), explain(2, 2, 2), bar(),
        hbars(300, 7),
        _cols(
            _stack(sub("30%"), control(label=128), explain(2, 4, 2), hbars(640, 19)),
            _stack(sub("28%"), explain(2, 3), vbars(320, 7), sub("38%"), control(80, 210), explain(2, 2, 2), vbars(320, 4)),
        ),
        FOOT,
    ],
    "Data preparation": [
        h1("18%"), intro("95%", "54%"),
        _stack(
            f'<div class="sk-tabs">{_bone(84, 14)}{_bone(100, 14)}</div>',
            control(label=72), explain(2),
            _cols(*[_stack(_bone(84, 12), _bone(96, 30, 8), gap=30) for _ in range(3)]),
            explain(2), hbars(120, 2), explain(2, 1), table(408, 11, 2),
        ),
        FOOT,
    ],
    "Models": [
        h1("38%"), intro("93%", "60%"), bar(), bar(), explain(2, 2),
        sub("26%"), table(185, 4, 8), explain(1, 2, 1),
        sub("28%"), control(label=40), explain(2, 1),
        f'<div class="sk-code">{"".join(_line(w, 18, 9) for w in ["52%", "0", "46%", "46%", "0", "46%", "46%", "46%"])}</div>',
        FOOT,
    ],
    "Evaluation": [
        h1("42%"), intro("96%", "64%"),
        sub("11%"), explain(2, 1, 2), curves(440),
        sub("16%"), explain(2, 1), vbars(340, 6),
        sub("23%"), explain(3, 1),
        _cols(_stack(matrix(), matrix()), _stack(matrix(), matrix())),
        sub("26%"), explain(2, 1, 2), hbars(510, 15), FOOT,
    ],
    "Findings": [
        h1("26%"), intro("72%"), _line("34%", 21), _line("78%", 21), _line("52%", 21),
        sub("16%"),
        f'<div class="sk-list" style="height:127px">{"".join(_line(w, 21) for w in ["88%", "76%", "62%", "81%"])}</div>',
        explain(2), caption("74%"), button(236), FOOT,
    ],
}


def _page(view):
    return _stack(*PAGES[view], gap=24)


def boot_html(views, first_view):
    """A full-page overlay: the real layout with bones, on the very first paint."""
    main = f'<div class="sk-main">{_stack(_header(), _kpis(), _page(first_view), gap=24)}</div>'
    return f'<div class="cs-boot" aria-hidden="true">{_sidebar(views)}{main}</div>'


def page_skeletons_html(views):
    """One hidden skeleton per page, shown in place of the page body when it is slow."""
    pages = "".join(f'<div class="cs-sk-page" data-view="{slug(v)}">{_page(v)}</div>' for v in views)
    return f'<div class="cs-sk-layer" aria-hidden="true">{pages}</div>'


def motion_css():
    return (HERE / "motion.css").read_text()


def motion_script():
    js = (HERE / "motion.js").read_text()
    # st.html sanitises with DOMPurify, which deletes a script if its text looks
    # like it contains markup. Fail loudly here rather than silently losing motion.
    found = re.search(r"<[/\w!]", js)
    if found:
        raise ValueError(f"motion.js must not contain {found.group()!r}; add a space after '<'.")
    return f"<script>{js}</script>"


def render_marker(view, run):
    """The last element of every script run. The browser reveals the page once it arrives."""
    return f'<div id="cs-rendered" data-view="{slug(view)}" data-run="{run}"></div>'
