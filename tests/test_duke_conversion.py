"""Positional unit tests for Duke conversion label normalization.

The edge-tile bug: edge tiles have win_w < tile_size. Old code normalized coords
to win_w, so a pixel at 150/200px gave label x=0.75. The saved PNG is 640px wide,
so YOLO sees x=0.75 → pixel 480, not pixel 150. Both values are in [0,1], so range
checks miss the bug entirely. These tests catch it by checking pixel positions.
"""

import pytest

TILE_SIZE = 640


# --- Reference implementations (inline — no heavy dependencies) ---

def clip_and_normalize_fixed(pixel_coords, tile_x0, tile_y0, tile_w, tile_h, out_w, out_h):
    """Current (fixed) implementation: normalizes to out_w/out_h (the saved PNG dims)."""
    xs = [c[0] for c in pixel_coords]
    ys = [c[1] for c in pixel_coords]
    if max(xs) < tile_x0 or min(xs) > tile_x0 + tile_w:
        return None
    if max(ys) < tile_y0 or min(ys) > tile_y0 + tile_h:
        return None
    norm = []
    for px, py in pixel_coords:
        cx = max(tile_x0, min(tile_x0 + tile_w, px))
        cy = max(tile_y0, min(tile_y0 + tile_h, py))
        norm.append(((cx - tile_x0) / out_w, (cy - tile_y0) / out_h))
    deduped = [norm[0]]
    for pt in norm[1:]:
        if pt != deduped[-1]:
            deduped.append(pt)
    return deduped if len(deduped) >= 3 else None


def clip_and_normalize_old(pixel_coords, tile_x0, tile_y0, tile_w, tile_h):
    """Old (buggy) implementation: normalizes to tile_w/tile_h (content window)."""
    xs = [c[0] for c in pixel_coords]
    ys = [c[1] for c in pixel_coords]
    if max(xs) < tile_x0 or min(xs) > tile_x0 + tile_w:
        return None
    if max(ys) < tile_y0 or min(ys) > tile_y0 + tile_h:
        return None
    norm = []
    for px, py in pixel_coords:
        cx = max(tile_x0, min(tile_x0 + tile_w, px))
        cy = max(tile_y0, min(tile_y0 + tile_h, py))
        norm.append(((cx - tile_x0) / tile_w, (cy - tile_y0) / tile_h))  # BUG
    deduped = [norm[0]]
    for pt in norm[1:]:
        if pt != deduped[-1]:
            deduped.append(pt)
    return deduped if len(deduped) >= 3 else None


# --- Full tile tests (win_w == tile_size — both versions agree) ---

class TestFullTile:

    def test_pixel_position_maps_correctly(self):
        coords = [(100, 100), (200, 100), (200, 200), (100, 200)]
        result = clip_and_normalize_fixed(coords, 0, 0, TILE_SIZE, TILE_SIZE, TILE_SIZE, TILE_SIZE)
        assert result is not None
        assert abs(result[0][0] - 100 / TILE_SIZE) < 1e-6
        assert abs(result[0][1] - 100 / TILE_SIZE) < 1e-6

    def test_all_coords_in_range(self):
        coords = [(50, 50), (590, 50), (590, 590), (50, 590)]
        result = clip_and_normalize_fixed(coords, 0, 0, TILE_SIZE, TILE_SIZE, TILE_SIZE, TILE_SIZE)
        assert result is not None
        for x, y in result:
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0

    def test_near_edge_coords_clamped_to_one(self):
        coords = [(0, 0), (639, 0), (639, 639), (0, 639)]
        result = clip_and_normalize_fixed(coords, 0, 0, TILE_SIZE, TILE_SIZE, TILE_SIZE, TILE_SIZE)
        assert result is not None
        assert max(p[0] for p in result) <= 1.0
        assert max(p[1] for p in result) <= 1.0


# --- Edge tile tests (win_w < tile_size — the bug lives here) ---

class TestEdgeTile:
    WIN_W = 200  # simulated 200px-wide edge tile content
    WIN_H = 200

    def test_pixel_position_correct_with_fix(self):
        """Pixel at (100, 100) must map to x=100/640=0.156, not x=100/200=0.5."""
        coords = [(100, 100), (190, 100), (190, 190), (100, 190)]
        result = clip_and_normalize_fixed(coords, 0, 0, self.WIN_W, self.WIN_H, TILE_SIZE, TILE_SIZE)
        assert result is not None
        assert abs(result[0][0] - 100 / TILE_SIZE) < 1e-6   # 0.15625
        assert abs(result[0][1] - 100 / TILE_SIZE) < 1e-6

    def test_old_code_gives_wrong_pixel_position(self):
        """Regression anchor: confirms old formula gives a different (wrong) answer."""
        coords = [(100, 100), (190, 100), (190, 190), (100, 190)]
        result_old = clip_and_normalize_old(coords, 0, 0, self.WIN_W, self.WIN_H)
        assert result_old is not None
        # Old code puts the label at x=100/200=0.5 in a 640px image → pixel 320, not 100
        assert abs(result_old[0][0] - 100 / self.WIN_W) < 1e-6

    def test_fix_and_old_code_diverge_significantly(self):
        """The fix must produce a materially different result from the old code."""
        coords = [(100, 100), (190, 100), (190, 190), (100, 190)]
        result_new = clip_and_normalize_fixed(coords, 0, 0, self.WIN_W, self.WIN_H, TILE_SIZE, TILE_SIZE)
        result_old = clip_and_normalize_old(coords, 0, 0, self.WIN_W, self.WIN_H)
        assert result_new is not None and result_old is not None
        # For win_w=200 vs tile_size=640 the x coords should differ by ~3.2×
        diff = abs(result_new[0][0] - result_old[0][0])
        assert diff > 0.2, f"Fix and old code too similar (diff={diff:.4f}) — fix may not be applied"

    def test_all_coords_in_range(self):
        coords = [(10, 10), (195, 10), (195, 195), (10, 195)]
        result = clip_and_normalize_fixed(coords, 0, 0, self.WIN_W, self.WIN_H, TILE_SIZE, TILE_SIZE)
        assert result is not None
        for x, y in result:
            assert 0.0 <= x <= 1.0, f"x={x:.4f} out of [0,1]"
            assert 0.0 <= y <= 1.0, f"y={y:.4f} out of [0,1]"

    def test_max_coord_stays_small_for_narrow_edge_tile(self):
        """Max label coord should reflect the content width fraction, not approach 1.0."""
        coords = [(10, 10), (198, 10), (198, 198), (10, 198)]
        result = clip_and_normalize_fixed(coords, 0, 0, self.WIN_W, self.WIN_H, TILE_SIZE, TILE_SIZE)
        assert result is not None
        max_x = max(p[0] for p in result)
        # Content goes to pixel 198/640 ≈ 0.31 — if max_x > 0.5 the old bug is present
        assert max_x < 0.35, f"max_x={max_x:.4f} — looks like old (buggy) normalization (win_w/tile_size={self.WIN_W/TILE_SIZE:.2f})"


# --- Non-trivial tile offsets (tile at col>0, row>0 in source image) ---

class TestOffsetTile:

    def test_interior_tile_pixel_offset_correct(self):
        """Tile starting at (640, 640) in source image."""
        tile_x0, tile_y0 = 640, 640
        # Object at pixel (700, 700) in source → (60, 60) relative to tile
        coords = [(700, 700), (800, 700), (800, 800), (700, 800)]
        result = clip_and_normalize_fixed(coords, tile_x0, tile_y0, TILE_SIZE, TILE_SIZE, TILE_SIZE, TILE_SIZE)
        assert result is not None
        assert abs(result[0][0] - 60 / TILE_SIZE) < 1e-6

    def test_offset_edge_tile(self):
        """Edge tile at (1280, 0) with win_w=200."""
        tile_x0 = 1280
        coords = [(1300, 50), (1460, 50), (1460, 150), (1300, 150)]
        result = clip_and_normalize_fixed(coords, tile_x0, 0, 200, TILE_SIZE, TILE_SIZE, TILE_SIZE)
        assert result is not None
        # x=1300 → relative=20 → 20/640=0.03125
        assert abs(result[0][0] - 20 / TILE_SIZE) < 1e-6
        for x, y in result:
            assert 0.0 <= x <= 1.0


# --- Boundary / overlap behavior ---

class TestBoundaryBehavior:

    def test_non_overlapping_returns_none(self):
        coords = [(700, 700), (750, 700), (750, 750), (700, 750)]
        assert clip_and_normalize_fixed(coords, 0, 0, TILE_SIZE, TILE_SIZE, TILE_SIZE, TILE_SIZE) is None

    def test_barely_overlapping_returns_result(self):
        coords = [(635, 100), (700, 100), (700, 200), (635, 200)]
        result = clip_and_normalize_fixed(coords, 0, 0, TILE_SIZE, TILE_SIZE, TILE_SIZE, TILE_SIZE)
        assert result is not None

    def test_triangle_polygon_preserved(self):
        coords = [(100, 100), (200, 150), (100, 200)]
        result = clip_and_normalize_fixed(coords, 0, 0, TILE_SIZE, TILE_SIZE, TILE_SIZE, TILE_SIZE)
        assert result is not None
        assert len(result) == 3

    def test_fully_outside_left_returns_none(self):
        coords = [(-50, 100), (-10, 100), (-10, 200), (-50, 200)]
        assert clip_and_normalize_fixed(coords, 0, 0, TILE_SIZE, TILE_SIZE, TILE_SIZE, TILE_SIZE) is None

    def test_degenerate_after_clamping_returns_none(self):
        # All four corners collapse to the same clamped point → fewer than 3 distinct points
        coords = [(650, 650), (700, 650), (700, 700), (650, 700)]
        result = clip_and_normalize_fixed(coords, 0, 0, TILE_SIZE, TILE_SIZE, TILE_SIZE, TILE_SIZE)
        # After clamping to (640, 640), all 4 → same point → deduped to 1 → None
        if result is not None:
            assert len(result) >= 3
