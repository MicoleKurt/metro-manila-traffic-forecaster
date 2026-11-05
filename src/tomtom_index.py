"""Scrape the real 2024-2025 Manila statistics that the public TomTom Traffic Index
page (tomtom.com/traffic-index/city/manila/) embeds in its HTML.

Data © TomTom. Collected from billions of anonymised driver trips across Metro
Manila. Keys used by TomTom: c = congestion %, v = speed km/h, fv = free-flow
speed km/h.
"""
import html
import json
import re
import urllib.request
from pathlib import Path

URL = "https://www.tomtom.com/traffic-index/city/{city}/"
RAW = Path("data/raw/tomtom")


def _decode(x):
    """Astro serialises props as [type, value]; 0 = plain, 1 = array."""
    if isinstance(x, list) and len(x) == 2 and isinstance(x[0], int):
        t, v = x
        if t == 1:
            return [_decode(i) for i in v]
        return _decode(v)
    if isinstance(x, dict):
        return {k: _decode(v) for k, v in x.items()}
    return x


def fetch(city="manila", refresh=False) -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    p = RAW / f"{city}.html"
    if refresh or not p.exists():
        req = urllib.request.Request(URL.format(city=city), headers={"User-Agent": "Mozilla/5.0"})
        p.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    page = p.read_text(encoding="utf-8")
    for m in re.finditer(r'props="([^"]*)"', page):
        props = json.loads(html.unescape(m.group(1)))
        if "traffic" in props:
            return _decode(props)["traffic"]
    raise ValueError("traffic data not found in page")


if __name__ == "__main__":
    t = fetch(refresh=True)

    def show(d, pre="", depth=0):
        for k, v in d.items():
            if isinstance(v, dict) and depth < 4:
                show(v, pre + k + ".", depth + 1)
            else:
                s = json.dumps(v)
                print(pre + k, "=", s[:160] + ("..." if len(s) > 160 else ""))
    show(t)
