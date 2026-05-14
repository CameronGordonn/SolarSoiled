# SolarSoiled Landing Page

AI-powered solar panel soiling detection — landing page source.

## Folder Structure

~~~
solarsoiled-landing/
  index.html
  styles.css
  assets/
    logo-placeholder.svg
    hero-placeholder.svg
    solution-placeholder.svg
    icon-upload.svg
    icon-segment.svg
    icon-map.svg
    icon-report.svg
    icon-heatmap.svg
    icon-csv.svg
    icon-pdf.svg
    icon-calendar.svg
  README.md
~~~

## Quick Start

To package the current checked-in site into `solarsoiled-landing.zip`:

~~~bash
python3 build_solarsoiled.py
~~~

### Option 1 — Python (no install needed)

~~~bash
cd solarsoiled-landing
python -m http.server 8000
~~~

Open http://localhost:8000

### Option 2 — Node

~~~bash
npx serve solarsoiled-landing
~~~

### Option 3 — Double-click

Open index.html directly in any browser. Fonts fall back to system-ui offline.

## Replacing Placeholder Images

All images in assets/ are labeled SVGs. To swap in real images:

1. Drop your files (PNG, JPG, WebP) into assets/.
2. Update the src attribute in index.html.
3. Recommended sizes:
   - Hero image: 800 x 500 px
   - Solution image: 600 x 400 px
   - Icons: 48 x 48 px (SVG preferred)
   - Logo: 32 x 32 px

## Color Palette

| Token     | Hex       | Usage                  |
|-----------|-----------|------------------------|
| Navy      | #0F2027   | Backgrounds, text      |
| Mid       | #203A43   | Gradient midpoint      |
| Teal      | #2C5364   | Accents, headings      |
| Gold      | #F7B731   | CTAs, highlights, tags |
| White     | #FFFFFF   | Cards, text on dark    |
| Off-white | #F8F9FA   | Alternating sections   |

## Fonts

- Headings: Space Grotesk (500, 700)
- Body: Inter (400, 500, 600)

Loaded via Google Fonts CDN. Offline fallback: system-ui, sans-serif.

## License

Internal project — not for redistribution.
