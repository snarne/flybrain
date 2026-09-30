"""Small SVG charts for the reports (no plotting packages needed)."""


def _head(w, h, title):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="sans-serif" font-size="12">',
            f'<rect width="{w}" height="{h}" fill="white"/>', f'<text x="10" y="18" font-size="14">{title}</text>']


def bars(items, title, w=640):
    """items: [(label, value)] as horizontal bars."""
    h = 40 + 22 * len(items)
    vmax = max(v for _, v in items) or 1
    lw = min(420, 10 + 7 * max(len(lab) for lab, _ in items))      # room for the labels
    w = max(w, lw + 260)
    o = _head(w, h, title)
    for i, (lab, v) in enumerate(items):
        y = 30 + 22 * i
        bw = (w - lw - 60) * v / vmax
        o.append(f'<text x="{lw}" y="{y + 14}" text-anchor="end">{lab}</text>')
        o.append(f'<rect x="{lw + 6}" y="{y + 3}" width="{bw:.1f}" height="15" fill="#1f77b4"/>')
        o.append(f'<text x="{lw + 10 + bw:.1f}" y="{y + 15}">{v:g}</text>')
    o.append("</svg>")
    return "\n".join(o)


def scatter(pts, xlab, ylab, title, w=520, h=420):
    """pts: [(x, y, failed)]."""
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    x1, y1 = (x1 if x1 > x0 else x0 + 1), (y1 if y1 > y0 else y0 + 1)
    L, R, T, B = 60, 20, 30, 45
    X = lambda v: L + (v - x0) / (x1 - x0) * (w - L - R)   # noqa: E731
    Y = lambda v: h - B - (v - y0) / (y1 - y0) * (h - T - B)   # noqa: E731
    o = _head(w, h, title)
    o += [f'<line x1="{L}" y1="{h - B}" x2="{w - R}" y2="{h - B}" stroke="#333"/>',
          f'<line x1="{L}" y1="{T}" x2="{L}" y2="{h - B}" stroke="#333"/>',
          f'<text x="{(L + w - R) / 2}" y="{h - 8}" text-anchor="middle">{xlab}</text>',
          f'<text x="14" y="{(T + h - B) / 2}" transform="rotate(-90 14 {(T + h - B) / 2})" text-anchor="middle">{ylab}</text>']
    for k in range(5):
        v = x0 + (x1 - x0) * k / 4
        o.append(f'<text x="{X(v):.1f}" y="{h - B + 16}" text-anchor="middle">{v:.3g}</text>')
        v = y0 + (y1 - y0) * k / 4
        o.append(f'<text x="{L - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">{v:.3g}</text>')
    for x, y, f in sorted(pts, key=lambda p: p[2]):
        o.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3" fill="{"#d62728" if f else "#9ecae1"}" '
                 f'fill-opacity="{0.9 if f else 0.6}"/>')
    o.append("</svg>")
    return "\n".join(o)
