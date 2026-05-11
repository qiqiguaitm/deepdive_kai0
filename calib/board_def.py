"""ChArUco 标定板公共定义.

设计:
  - 模块顶层常量 (COLS / ROWS / SQUARE_MM / MARKER_MM / DICT_ID) = **默认板**
    (7×5, sq 38mm, mk 28mm, DICT_5X5_100). 仅 capture_handeye / solve_calibration
    等老脚本依赖, 不影响新的 intrinsic pipeline.
  - 新代码统一用 `BoardSpec` + `get_board(spec)` / `detect_charuco(image, K, dist,
    spec=None)` 接口, 允许任意板.
  - 默认值通过 `BoardSpec.default()` 暴露; YAML 加载用 `BoardSpec.from_yaml(path)`.

新板 (9×15 等) 用法:
    spec = BoardSpec(cols=9, rows=15, square_mm=30.0, marker_mm=22.0,
                     dict_id=cv2.aruco.DICT_5X5_100)
    board = get_board(spec)
    detect_charuco(img, K, dist, spec)
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── 默认板规格 (向后兼容旧 capture_handeye / solve_calibration) ─────────────
COLS = 7
ROWS = 5
SQUARE_MM = 38.0
MARKER_MM = 28.0
DICT_ID = cv2.aruco.DICT_5X5_100

BOARD_W_MM = COLS * SQUARE_MM
BOARD_H_MM = ROWS * SQUARE_MM
N_CORNERS = (COLS - 1) * (ROWS - 1)
N_MARKERS = (COLS * ROWS) // 2
SQUARE_M = SQUARE_MM / 1000.0
MARKER_M = MARKER_MM / 1000.0


# ── 通用板描述 ──────────────────────────────────────────────────────────────

# OpenCV ArUco dict name → enum id; covers all predefined dicts up to OpenCV 4.10.
_DICT_NAME_MAP: dict[str, int] = {
    name: getattr(cv2.aruco, name)
    for name in (
        "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
        "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
        "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
        "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
        "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9",
        "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11",
    ) if hasattr(cv2.aruco, name)
}


def dict_id_from_name(name: str) -> int:
    if name in _DICT_NAME_MAP:
        return _DICT_NAME_MAP[name]
    raise KeyError(f"unknown ArUco dict {name!r}; supported: {sorted(_DICT_NAME_MAP)}")


def dict_name_from_id(did: int) -> str:
    for n, v in _DICT_NAME_MAP.items():
        if v == did:
            return n
    return f"unknown({did})"


@dataclasses.dataclass(frozen=True)
class BoardSpec:
    """All info needed to instantiate a CharucoBoard + name it for logging."""
    cols: int            # squaresX (OpenCV: number of squares along X)
    rows: int            # squaresY
    square_mm: float
    marker_mm: float
    dict_id: int
    # OpenCV 4.6+ changed the default marker-on-board layout. Boards printed
    # with older OpenCV / Calib.io's legacy generator need this enabled so
    # CharucoDetector's geometry template matches. Symptom when wrong: many
    # ArUco markers are detected per detectMarkers(), but
    # CharucoDetector.detectBoard() returns 0 charuco corners — the markers
    # are visible but their IDs are in unexpected board cells.
    legacy_pattern: bool = False

    @property
    def square_m(self) -> float: return self.square_mm / 1000.0

    @property
    def marker_m(self) -> float: return self.marker_mm / 1000.0

    @property
    def n_corners(self) -> int:
        """ChArUco *interior* chessboard corners — (cols-1)*(rows-1)."""
        return (self.cols - 1) * (self.rows - 1)

    @property
    def n_markers(self) -> int:
        """How many ArUco markers the board carries (half of squares)."""
        return (self.cols * self.rows) // 2

    @property
    def dict_name(self) -> str:
        return dict_name_from_id(self.dict_id)

    def __str__(self) -> str:
        return (f"BoardSpec({self.cols}×{self.rows}, sq={self.square_mm}mm, "
                f"mk={self.marker_mm}mm, {self.dict_name}, "
                f"n_corners={self.n_corners}, n_markers={self.n_markers})")

    # ── factories ─────────────────────────────────────────────────────────
    @classmethod
    def default(cls) -> "BoardSpec":
        return cls(cols=COLS, rows=ROWS, square_mm=SQUARE_MM, marker_mm=MARKER_MM,
                   dict_id=DICT_ID)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BoardSpec":
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(
            cols=int(d["cols"]), rows=int(d["rows"]),
            square_mm=float(d["square_mm"]), marker_mm=float(d["marker_mm"]),
            dict_id=dict_id_from_name(str(d["dict"])),
            legacy_pattern=bool(d.get("legacy_pattern", False)),
        )

    @classmethod
    def from_cli(cls, args, fallback: Optional["BoardSpec"] = None) -> "BoardSpec":
        """Build from argparse Namespace. Supports --board-config (YAML) +
        inline --board-cols / --board-rows / --board-square-mm /
        --board-marker-mm / --board-dict overrides (latter take precedence).
        Falls back to `fallback` or `default()`."""
        if getattr(args, "board_config", None):
            spec = cls.from_yaml(args.board_config)
        else:
            spec = fallback if fallback is not None else cls.default()
        cols = getattr(args, "board_cols", None) or spec.cols
        rows = getattr(args, "board_rows", None) or spec.rows
        sqmm = getattr(args, "board_square_mm", None) or spec.square_mm
        mkmm = getattr(args, "board_marker_mm", None) or spec.marker_mm
        dname = getattr(args, "board_dict", None)
        did = dict_id_from_name(dname) if dname else spec.dict_id
        legacy = getattr(args, "board_legacy", None)
        legacy = spec.legacy_pattern if legacy is None else bool(legacy)
        return cls(cols=int(cols), rows=int(rows),
                   square_mm=float(sqmm), marker_mm=float(mkmm), dict_id=did,
                   legacy_pattern=legacy)


def add_board_cli_args(parser):
    """Mount --board-* flags on an argparse parser. Default values are None →
    treated as 'use spec from --board-config or built-in default'."""
    g = parser.add_argument_group("Board specs (default: 7×5, 38mm/28mm, DICT_5X5_100)")
    g.add_argument("--board-config", default=None,
                   help="YAML with keys: cols, rows, square_mm, marker_mm, dict")
    g.add_argument("--board-cols", type=int, default=None,
                   help="squares along X (e.g. 9 for a 9×15 board)")
    g.add_argument("--board-rows", type=int, default=None,
                   help="squares along Y (e.g. 15)")
    g.add_argument("--board-square-mm", type=float, default=None)
    g.add_argument("--board-marker-mm", type=float, default=None)
    g.add_argument("--board-dict", default=None,
                   help="ArUco dict name (e.g. DICT_5X5_100, DICT_5X5_250)")
    g.add_argument("--board-legacy", action="store_true", default=None,
                   help=("enable CharucoBoard.setLegacyPattern(True) — many "
                         "third-party / Calib.io boards printed before OpenCV "
                         "4.6 use the legacy marker layout. If markers are "
                         "detected but charuco corners stay at 0, try this."))
    return parser


# ── Detector / board / detect helpers ──────────────────────────────────────

def get_dictionary(spec: Optional[BoardSpec] = None) -> cv2.aruco.Dictionary:
    spec = spec or BoardSpec.default()
    return cv2.aruco.getPredefinedDictionary(spec.dict_id)


def get_board(spec: Optional[BoardSpec] = None) -> cv2.aruco.CharucoBoard:
    """Returns a CharucoBoard instance (units: meters)."""
    spec = spec or BoardSpec.default()
    board = cv2.aruco.CharucoBoard(
        size=(spec.cols, spec.rows),
        squareLength=spec.square_m,
        markerLength=spec.marker_m,
        dictionary=get_dictionary(spec),
    )
    if spec.legacy_pattern and hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(True)
    return board


def get_detector_params(refine_markers: bool = False) -> cv2.aruco.DetectorParameters:
    """Tuned ArUco detector params.

    refine_markers (default False) — whether to do sub-pixel refinement on
    ArUco marker corners. For *ChArUco detection* the OpenCV Chinese tutorial
    (fengzhenHIT/OpenCV-contrib-module-Chinese-Tutorials, chapter 2) and the
    official OpenCV docs explicitly recommend disabling marker-side refinement
    because the subpixel pass can shift marker corners off the integer grid
    and those errors then propagate into the interpolated ChArUco corners.

    For *standalone ArUco pose estimation* (no chessboard interpolation) leave
    refinement on by passing refine_markers=True — those workflows benefit.

    Capture/solve/verify intrinsics scripts call this without args → no refine."""
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 10
    if refine_markers:
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 30
        params.cornerRefinementMinAccuracy = 0.01
    else:
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    return params


def detect_charuco(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    spec: Optional[BoardSpec] = None,
) -> tuple[np.ndarray | None, np.ndarray | None,
           np.ndarray | None, np.ndarray | None,
           float | None, int]:
    """Detect ChArUco corners + estimate board pose.

    Returns (charuco_corners, charuco_ids, rvec, tvec, reprojection_error,
             n_aruco_markers). Detection failure → first 5 None, last 0.
    """
    spec = spec or BoardSpec.default()
    board = get_board(spec)
    params = get_detector_params()
    detector = cv2.aruco.ArucoDetector(get_dictionary(spec), params)

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    marker_corners, marker_ids, _ = detector.detectMarkers(gray)
    if marker_ids is None or len(marker_ids) < 2:
        n_found = 0 if marker_ids is None else len(marker_ids)
        return None, None, None, None, None, n_found

    n_aruco_markers = len(marker_ids)

    charuco_detector = cv2.aruco.CharucoDetector(board, detectorParams=params)
    charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)
    if charuco_ids is None or len(charuco_ids) < 6:
        return None, None, None, None, None, n_aruco_markers

    obj_pts = board.getChessboardCorners()[charuco_ids.flatten()]
    success, rvec, tvec = cv2.solvePnP(
        obj_pts, charuco_corners, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return charuco_corners, charuco_ids, None, None, None, n_aruco_markers

    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
    if obj_points is not None and len(obj_points) > 0:
        projected, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
        err = np.sqrt(np.mean((img_points - projected.reshape(-1, 1, 2)) ** 2))
    else:
        err = None

    return charuco_corners, charuco_ids, rvec, tvec, err, n_aruco_markers
