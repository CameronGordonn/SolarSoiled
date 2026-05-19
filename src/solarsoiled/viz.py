"""Folium risk map generator.

Reads risk.geojson (WGS84, one polygon per detected array with a risk_score
field [0.0, 1.0]) and writes a single self-contained HTML file with an
interactive map — arrays colored green→yellow→red by soiling risk on a
satellite basemap, with hover tooltips per array.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

BASEMAP_TILES = {
    "satellite": {
        "tiles": (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        "attr": (
            "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, "
            "GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
        ),
        "name": "ESRI Satellite",
    },
    "osm": {
        "tiles": "OpenStreetMap",
        "attr": "",
        "name": "OpenStreetMap",
    },
    "topo": {
        "tiles": (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
        ),
        "attr": (
            "Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ, TomTom, Intermap, "
            "iPC, USGS, FAO, NPS, NRCAN, GeoBase, Kadaster NL, Ordnance Survey, "
            "Esri Japan, METI, Esri China (Hong Kong), and the GIS User Community"
        ),
        "name": "ESRI Topo",
    },
}

_RISK_COLORS = ["#22c55e", "#eab308", "#ef4444"]  # green → yellow → red
_UNSCORED_COLOR = "#94a3b8"  # slate-gray for arrays without a score


def build_risk_map(
    risk_geojson: Path,
    out_html: Path,
    *,
    recommendations_json: Path | None = None,
    basemap: str = "satellite",
) -> Path:
    """Render risk.geojson as an interactive Folium map, write to out_html.

    Returns out_html path.
    """
    try:
        import folium
        from branca.colormap import LinearColormap
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            "folium and geopandas are required. Run: pip install folium"
        ) from exc

    gdf = gpd.read_file(risk_geojson)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    has_risk = "risk_score" in gdf.columns
    if not has_risk:
        warnings.warn(
            "risk.geojson has no risk_score column — all arrays will render gray. "
            "Run `solarsoiled score` to add soiling risk scores.",
            stacklevel=2,
        )

    # Optionally join cleaning windows from recommendations.json
    rec_by_id: dict[int, str] = {}
    if recommendations_json and Path(recommendations_json).exists():
        try:
            raw = json.loads(Path(recommendations_json).read_text())
            entries = raw if isinstance(raw, list) else raw.get("arrays", [])
            for e in entries:
                aid = e.get("array_id")
                window = e.get("cleaning_window") or (e.get("recommendation") or {}).get("window")
                if aid is not None and window:
                    rec_by_id[int(aid)] = str(window)
        except Exception:
            pass  # recommendations are optional — never crash the map render

    has_recs = bool(rec_by_id)
    if has_recs:
        gdf["cleaning_window"] = gdf["array_id"].map(rec_by_id).fillna("—")

    # Round display columns
    if has_risk:
        gdf["risk_score"] = gdf["risk_score"].round(3)
    if "area_m2" in gdf.columns:
        gdf["area_m2"] = gdf["area_m2"].round(1)

    colormap = LinearColormap(_RISK_COLORS, vmin=0.0, vmax=1.0, caption="Soiling Risk Score")

    # Map center + init
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy] lon/lat
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    tile_cfg = BASEMAP_TILES.get(basemap, BASEMAP_TILES["satellite"])
    if basemap == "osm":
        m = folium.Map(location=center, zoom_start=16)
    else:
        m = folium.Map(location=center, zoom_start=16, tiles=None)
        folium.TileLayer(
            tiles=tile_cfg["tiles"],
            attr=tile_cfg["attr"],
            name=tile_cfg["name"],
            max_zoom=20,
        ).add_to(m)

    def _style(feature):
        score = feature["properties"].get("risk_score")
        fill = colormap(float(score)) if (score is not None and has_risk) else _UNSCORED_COLOR
        return {"fillColor": fill, "fillOpacity": 0.65, "color": "#1e293b", "weight": 1.5}

    # Build tooltip from whichever columns are present
    tip_fields, tip_aliases = [], []
    for col, alias in [
        ("array_id", "Array ID"),
        ("risk_score", "Risk Score"),
        ("area_m2", "Area (m²)"),
        ("cleaning_window", "Clean by"),
    ]:
        if col in gdf.columns:
            tip_fields.append(col)
            tip_aliases.append(alias)

    folium.GeoJson(
        gdf,
        name="Solar Arrays",
        style_function=_style,
        tooltip=folium.GeoJsonTooltip(
            fields=tip_fields,
            aliases=tip_aliases,
            localize=True,
            sticky=False,
        ) if tip_fields else None,
    ).add_to(m)

    if has_risk:
        colormap.add_to(m)

    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    folium.LayerControl().add_to(m)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))
    return out_html
