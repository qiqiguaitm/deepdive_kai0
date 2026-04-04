"""
ChArUco 标定板公共定义

所有标定模块统一 import 此文件，避免参数重复定义。
板规格: 7×5, sq=38mm, mk=28mm, DICT_5X5_100, A4 横版
"""
import cv2
import numpy as np

# ── 标定板参数 ──────────────────────────────────────────────────────────────
COLS = 7               # X 方向 square 数 (列)
ROWS = 5               # Y 方向 square 数 (行)
SQUARE_MM = 38.0        # 棋盘格边长 (mm)
MARKER_MM = 28.0        # ArUco 标记边长 (mm)
DICT_ID = cv2.aruco.DICT_5X5_100

# 物理尺寸 (mm)
BOARD_W_MM = COLS * SQUARE_MM   # 266mm
BOARD_H_MM = ROWS * SQUARE_MM   # 190mm

# 角点 / 标记数量
N_CORNERS = (COLS - 1) * (ROWS - 1)   # 24
N_MARKERS = (COLS * ROWS) // 2          # 17

# 单位换算 (标定用 m, 打印用 mm)
SQUARE_M = SQUARE_MM / 1000.0
MARKER_M = MARKER_MM / 1000.0


def get_dictionary() -> cv2.aruco.Dictionary:
    """返回 ArUco 字典"""
    return cv2.aruco.getPredefinedDictionary(DICT_ID)


def get_board() -> cv2.aruco.CharucoBoard:
    """返回 CharucoBoard 实例 (单位: m)"""
    return cv2.aruco.CharucoBoard(
        size=(COLS, ROWS),
        squareLength=SQUARE_M,
        markerLength=MARKER_M,
        dictionary=get_dictionary(),
    )


def get_detector_params() -> cv2.aruco.DetectorParameters:
    """返回优化过的 ArUco 检测参数"""
    params = cv2.aruco.DetectorParameters()
    # 降低自适应阈值窗口，提升近距小 marker 检测率
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 10
    # 放宽角点精化参数
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 30
    params.cornerRefinementMinAccuracy = 0.01
    return params


def detect_charuco(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, float | None, int]:
    """
    检测 ChArUco 角点并估计板位姿。

    Args:
        image: BGR 或灰度图
        camera_matrix: 3×3 内参矩阵
        dist_coeffs: 畸变系数

    Returns:
        (charuco_corners, charuco_ids, rvec, tvec, reprojection_error, n_aruco_markers)
        检测失败时前 5 项返回 None, n_aruco_markers 返回 0
    """
    board = get_board()
    params = get_detector_params()
    detector = cv2.aruco.ArucoDetector(get_dictionary(), params)

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: 检测 ArUco markers
    marker_corners, marker_ids, _ = detector.detectMarkers(gray)
    if marker_ids is None or len(marker_ids) < 2:
        n_found = 0 if marker_ids is None else len(marker_ids)
        return None, None, None, None, None, n_found

    n_aruco_markers = len(marker_ids)

    # Step 2: 插值 ChArUco 角点 + 估计位姿
    charuco_detector = cv2.aruco.CharucoDetector(board, detectorParams=params)
    charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)
    if charuco_ids is None or len(charuco_ids) < 6:
        return None, None, None, None, None, n_aruco_markers

    # Step 3: 估计位姿
    obj_pts = board.getChessboardCorners()[charuco_ids.flatten()]
    success, rvec, tvec = cv2.solvePnP(
        obj_pts, charuco_corners, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return charuco_corners, charuco_ids, None, None, None, n_aruco_markers

    # Step 4: 计算重投影误差
    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
    if obj_points is not None and len(obj_points) > 0:
        projected, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
        err = np.sqrt(np.mean((img_points - projected.reshape(-1, 1, 2)) ** 2))
    else:
        err = None

    return charuco_corners, charuco_ids, rvec, tvec, err, n_aruco_markers
