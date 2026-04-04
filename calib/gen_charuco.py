#!/usr/bin/env python3
"""
ChArUco board PDF — A4 Landscape, exact physical dimensions.
Board: 7 cols × 5 rows, square=38mm, marker=28mm, DICT_5X5_100
Board area: 266 × 190 mm → fits A4 landscape (297 × 210 mm)
"""

import cv2
import numpy as np
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.colors import black, Color

from board_def import COLS, ROWS, SQUARE_MM, MARKER_MM, BOARD_W_MM, BOARD_H_MM, get_dictionary

PAGE_W, PAGE_H = landscape(A4)  # 841.89 × 595.28 pt = 297 × 210 mm

def create_pdf(output_path):
    c = canvas.Canvas(output_path, pagesize=landscape(A4))
    c.setTitle("ChArUco 7x5 sq38 mk28 DICT_5X5_100")

    # Generate board image at 600 DPI
    aruco_dict = get_dictionary()
    board = cv2.aruco.CharucoBoard(
        size=(COLS, ROWS),
        squareLength=SQUARE_MM / 1000.0,
        markerLength=MARKER_MM / 1000.0,
        dictionary=aruco_dict
    )
    px_per_mm = 600 / 25.4
    out_w = int(BOARD_W_MM * px_per_mm)
    out_h = int(BOARD_H_MM * px_per_mm)
    img = board.generateImage(outSize=(out_w, out_h), marginSize=0)

    import tempfile, os
    tmp_png = os.path.join(tempfile.gettempdir(), "charuco_temp.png")
    cv2.imwrite(tmp_png, img)

    # Board size in points
    bw = BOARD_W_MM * mm
    bh = BOARD_H_MM * mm

    # Center on landscape A4
    x0 = (PAGE_W - bw) / 2
    y0 = (PAGE_H - bh) / 2

    # Draw board
    c.drawImage(tmp_png, x0, y0, width=bw, height=bh)

    # Border
    c.setStrokeColor(Color(0.7, 0.7, 0.7))
    c.setLineWidth(0.3)
    c.rect(x0, y0, bw, bh, stroke=1, fill=0)

    # Top ruler (mm)
    c.setStrokeColor(black)
    c.setLineWidth(0.4)
    c.setFont("Helvetica", 5.5)
    for i in range(COLS + 1):
        x = x0 + i * SQUARE_MM * mm
        ty = y0 + bh + 2 * mm
        tl = 3 * mm if i == 0 or i == COLS else 2 * mm
        c.line(x, ty, x, ty + tl)
        c.drawCentredString(x, ty + tl + 1.5 * mm, f"{i * SQUARE_MM:.0f}")

    # Left ruler (mm)
    for i in range(ROWS + 1):
        y = y0 + i * SQUARE_MM * mm
        tx = x0 - 2 * mm
        tl = 3 * mm if i == 0 or i == ROWS else 2 * mm
        c.line(tx - tl, y, tx, y)
        c.drawRightString(tx - tl - 1 * mm, y - 2, f"{i * SQUARE_MM:.0f}")

    # 50mm verification square (bottom-right corner)
    vs = 50 * mm
    vx = PAGE_W - 15 * mm - vs
    vy = 8 * mm
    c.setLineWidth(0.4)
    c.setDash(2, 2)
    c.rect(vx, vy, vs, vs, stroke=1, fill=0)
    c.setDash()
    c.setFont("Helvetica", 5.5)
    c.drawCentredString(vx + vs / 2, vy - 3 * mm, "Verify: 50 × 50 mm")

    # Info
    c.setFont("Helvetica", 6.5)
    c.setFillColor(Color(0.3, 0.3, 0.3))
    c.drawCentredString(PAGE_W / 2, 10 * mm,
        f"ChArUco {COLS}×{ROWS}  |  Square {SQUARE_MM:.0f}mm  |  "
        f"Marker {MARKER_MM:.0f}mm  |  DICT_5X5_100  |  "
        f"Board {BOARD_W_MM:.0f}×{BOARD_H_MM:.0f}mm")

    # Warning
    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(Color(0.8, 0, 0))
    c.drawCentredString(PAGE_W / 2, PAGE_H - 8 * mm,
        "PRINT AT 100% ACTUAL SIZE — DO NOT FIT TO PAGE")

    try:
        c.save()
    finally:
        if os.path.exists(tmp_png):
            os.remove(tmp_png)
    print(f"Saved: {output_path}")
    print(f"Page:  A4 landscape (297 × 210 mm)")
    print(f"Board: {BOARD_W_MM:.0f} × {BOARD_H_MM:.0f} mm, centered")
    print(f"Margin: {(297 - BOARD_W_MM)/2:.1f}mm left/right, "
          f"{(210 - BOARD_H_MM)/2:.1f}mm top/bottom")


if __name__ == '__main__':
    create_pdf("charuco_7x5_sq38_landscape.pdf")

