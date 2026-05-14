"""Download the Duke/Bradbury solar array dataset (Figshare annotations + USGS NAIP imagery)."""

import argparse
import sys
import time
from pathlib import Path

try:
    import requests
    import pandas as pd
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError as e:
    print(f"Missing dependency: {e}. Run: pip install requests pandas")
    sys.exit(1)

FIGSHARE_ARTICLE_ID = 3385780
FIGSHARE_API_BASE = "https://api.figshare.com/v2"
NAIP_PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
ANNOTATION_EXTENSIONS = {".geojson", ".csv", ".json", ".txt", ".md"}


def build_http_session() -> requests.Session:
    retry = Retry(total=5, connect=5, read=5, status=5, backoff_factor=1.0,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset(["GET", "HEAD"]), raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "solar-soiling-ml/duke-downloader"})
    return session


HTTP = build_http_session()


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download_file(url: str, dest: Path, expected_size: int = 0) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected_size == 0 or dest.stat().st_size == expected_size):
        print(f"  [skip] {dest.name} already downloaded")
        return True
    try:
        with HTTP.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=256 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if expected_size:
                        print(f"\r  {human_size(downloaded)} / {human_size(expected_size)} ({downloaded/expected_size*100:.1f}%)", end="", flush=True)
            print()
        return True
    except Exception as e:
        print(f"\n  [error] {e}")
        if dest.exists():
            dest.unlink()
        return False


def fetch_figshare_files(article_id: int) -> list:
    resp = HTTP.get(f"{FIGSHARE_API_BASE}/articles/{article_id}/files", timeout=30)
    resp.raise_for_status()
    return resp.json()


def download_annotations(output_dir: Path, list_only: bool = False) -> None:
    print(f"Fetching Figshare file list (article {FIGSHARE_ARTICLE_ID})...")
    files = fetch_figshare_files(FIGSHARE_ARTICLE_ID)
    annotations = [f for f in files if Path(f["name"]).suffix.lower() in ANNOTATION_EXTENSIONS]
    other = [f for f in files if Path(f["name"]).suffix.lower() not in ANNOTATION_EXTENSIONS]
    print(f"\nAnnotation files: {len(annotations)} ({human_size(sum(f.get('size', 0) for f in annotations))})")
    for f in annotations:
        print(f"    {f['name']}  ({human_size(f.get('size', 0))})")
    print(f"  Ortho/imagery files: {len(other)} ({human_size(sum(f.get('size', 0) for f in other))})")
    if list_only:
        return
    print(f"\nDownloading {len(annotations)} annotation files to {output_dir}/")
    for f in annotations:
        print(f"  → {f['name']} ({human_size(f.get('size', 0))})")
        download_file(f["download_url"], output_dir / f["name"], expected_size=f.get("size", 0))
    print("\nAnnotation download complete.")


def query_naip_pc(bbox: tuple, datetime_range: str | None = None) -> list:
    try:
        body = {
            "collections": ["naip"], "bbox": list(bbox), "limit": 10,
            "sortby": [{"field": "datetime", "direction": "desc"}],
        }
        if datetime_range:
            body["datetime"] = datetime_range
        resp = HTTP.post(NAIP_PC_STAC, json=body, timeout=30)
        return [] if resp.status_code >= 400 else resp.json().get("features", [])
    except Exception:
        return []


def find_best_naip_scene(features: list, bbox_wgs84: tuple) -> tuple[dict | None, str]:
    import rasterio
    from rasterio.windows import from_bounds
    from pyproj import Transformer

    best_feature, best_url, best_overlap = None, "", 0.0
    for feature in features:
        url = feature.get("assets", {}).get("image", {}).get("href", "")
        if not url:
            continue
        try:
            with rasterio.open(url) as ds:
                t = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
                x0, y0 = t.transform(bbox_wgs84[0], bbox_wgs84[1])
                x1, y1 = t.transform(bbox_wgs84[2], bbox_wgs84[3])
                win = from_bounds(x0, y0, x1, y1, ds.transform)
                clipped_col = max(0.0, min(win.col_off + win.width, float(ds.width))) - max(0.0, win.col_off)
                clipped_row = max(0.0, min(win.row_off + win.height, float(ds.height))) - max(0.0, win.row_off)
                overlap = (clipped_col * clipped_row) / (win.width * win.height)
                if overlap > best_overlap:
                    best_overlap, best_feature, best_url = overlap, feature, url
                if overlap >= 0.999:
                    break
        except Exception:
            continue
    return best_feature, best_url


def download_naip_crop(image_url: str, bbox_wgs84: tuple, dest: Path) -> bool:
    import rasterio
    from rasterio.windows import from_bounds
    from pyproj import Transformer

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with rasterio.open(image_url) as src:
            t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            x0, y0 = t.transform(bbox_wgs84[0], bbox_wgs84[1])
            x1, y1 = t.transform(bbox_wgs84[2], bbox_wgs84[3])
            window = from_bounds(x0, y0, x1, y1, src.transform)
            out_h = max(1, int(round(window.height)))
            out_w = max(1, int(round(window.width)))
            n_bands = min(3, src.count)
            data = src.read(indexes=list(range(1, n_bands + 1)), window=window, out_shape=(n_bands, out_h, out_w))
            win_transform = src.window_transform(window)
        with rasterio.open(dest, "w", driver="GTiff", height=out_h, width=out_w, count=n_bands,
                           dtype=data.dtype, crs=src.crs, transform=win_transform, compress="lzw") as dst:
            dst.write(data)
        print(f"  saved {dest.name}  ({out_w}×{out_h} px, {dest.stat().st_size/1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f"  [crop error] {e}")
        if dest.exists():
            dest.unlink()
        return False


BRADBURY_VINTAGE = "2014-01-01T00:00:00Z/2015-12-31T23:59:59Z"


def download_imagery(meta_csv: Path, ortho_dir: Path, min_polygons: int = 0, limit: int = 0,
                     datetime_range: str | None = BRADBURY_VINTAGE) -> None:
    if not meta_csv.exists():
        raise FileNotFoundError(f"Annotation CSV not found: {meta_csv}\nRun --annotations-only first.")

    meta = pd.read_csv(meta_csv)
    imgs = meta.groupby("image_name").agg(
        city=("city", "first"), polygons=("polygon_id", "count"),
        nw_lat=("nw_corner_of_image_latitude", "first"), nw_lon=("nw_corner_of_image_longitude", "first"),
        se_lat=("se_corner_of_image_latitude", "first"), se_lon=("se_corner_of_image_longitude", "first"),
    ).reset_index()

    if min_polygons > 0:
        before = len(imgs)
        imgs = imgs[imgs["polygons"] >= min_polygons].reset_index(drop=True)
        print(f"Filtered to images with >= {min_polygons} polygons: {before} → {len(imgs)}")

    imgs = imgs.sort_values("polygons", ascending=False).reset_index(drop=True)
    if limit > 0:
        imgs = imgs.head(limit)
        print(f"--limit {limit}: downloading top {len(imgs)} images by polygon count")

    ortho_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nAttempting to download {len(imgs)} ortho images...\nOutput: {ortho_dir}/\n")

    failed, skipped = [], []
    for i, row in imgs.iterrows():
        img_name = row["image_name"]
        dest = ortho_dir / f"{img_name}.tif"
        if dest.exists():
            print(f"[{i+1}/{len(imgs)}] {img_name} — already exists, skipping")
            skipped.append(img_name)
            continue

        bbox = (row["nw_lon"], row["se_lat"], row["se_lon"], row["nw_lat"])
        print(f"[{i+1}/{len(imgs)}] {img_name} ({row['city']}, {row['polygons']} polygons)")
        print(f"  bbox: {bbox[0]:.5f},{bbox[1]:.5f} → {bbox[2]:.5f},{bbox[3]:.5f}")
        try:
            features = query_naip_pc(bbox, datetime_range=datetime_range)
            if not features and datetime_range:
                print(f"  [no 2014-2015 scenes for this bbox — falling back to any vintage]")
                features = query_naip_pc(bbox)
            if not features:
                print(f"  [no results] No NAIP scenes found")
                failed.append((img_name, "no NAIP scenes"))
                time.sleep(0.5)
                continue
            feature, image_url = find_best_naip_scene(features, bbox)
            if not feature:
                print(f"  [no match] {len(features)} scenes returned but none readable")
                failed.append((img_name, "no readable NAIP scene"))
                time.sleep(0.5)
                continue
            scene_id = feature.get("id", "?")
            scene_date = feature.get("properties", {}).get("datetime", "?")[:10]
            print(f"  NAIP: {scene_id}  ({scene_date})")
            if not download_naip_crop(image_url, bbox, dest):
                failed.append((img_name, "crop/download failed"))
            time.sleep(0.3)
        except Exception as e:
            print(f"  [error] {e}")
            failed.append((img_name, str(e)))
            time.sleep(1)

    downloaded = len(imgs) - len(failed) - len(skipped)
    print(f"\nDownloaded: {downloaded}  |  Skipped: {len(skipped)}  |  Failed: {len(failed)}")
    if failed:
        print(f"\nFailed images ({len(failed)}):")
        for name, reason in failed[:20]:
            print(f"  {name}: {reason}")
        if len(failed) > 20:
            print(f"  ... and {len(failed)-20} more")
    if downloaded > 0 or skipped:
        print("\nNext step:\n  python scripts/02d_convert_duke_dataset.py")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=repo_root / "data/raw/duke")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--annotations-only", action="store_true")
    parser.add_argument("--imagery-only", action="store_true")
    parser.add_argument("--min-polygons", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--datetime", type=str, default=BRADBURY_VINTAGE,
                        help="STAC datetime filter (default: 2014-2015 to match Bradbury annotations). "
                             "Pass 'any' to disable and pull the latest NAIP.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_only:
        download_annotations(args.output_dir, list_only=True)
        return
    if not args.imagery_only:
        download_annotations(args.output_dir)
    if not args.annotations_only:
        meta_csv = args.output_dir / "polygonDataExceptVertices.csv"
        ortho_dir = args.output_dir / "ortho"
        datetime_range = None if args.datetime.lower() == "any" else args.datetime
        download_imagery(meta_csv, ortho_dir, min_polygons=args.min_polygons, limit=args.limit,
                         datetime_range=datetime_range)


if __name__ == "__main__":
    main()
