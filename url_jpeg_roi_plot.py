import argparse
import collections
import time
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
import requests


WINDOW_NAME = "URL JPEG ROI Channel Mean (Press q to quit)"
FRAME_WIDTH = 960
FRAME_HEIGHT = 720
MAX_POINTS = 320
PLOT_HEIGHT = 220
VAR_Y_MAX_BASE = 4000.0

selecting = False
selection_start: Optional[Tuple[int, int]] = None
selection_end: Optional[Tuple[int, int]] = None
roi_rect: Optional[Tuple[int, int, int, int]] = None


def normalize_rect(p1: Tuple[int, int], p2: Tuple[int, int]) -> Tuple[int, int, int, int]:
    x1, y1 = p1
    x2, y2 = p2
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def on_mouse(event, x, y, _flags, _param):
    global selecting, selection_start, selection_end, roi_rect

    if event == cv2.EVENT_LBUTTONDOWN:
        selecting = True
        selection_start = (x, y)
        selection_end = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and selecting:
        selection_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and selecting:
        selecting = False
        selection_end = (x, y)
        if selection_start and selection_end:
            x1, y1, x2, y2 = normalize_rect(selection_start, selection_end)
            if x2 - x1 > 2 and y2 - y1 > 2:
                roi_rect = (x1, y1, x2, y2)


def draw_plot(
    canvas: np.ndarray,
    b_values: List[float],
    g_values: List[float],
    r_values: List[float],
    b_var_values: List[float],
    g_var_values: List[float],
    r_var_values: List[float],
    var_scale: float,
) -> None:
    h, w, _ = canvas.shape
    margin_left, margin_right, margin_top, margin_bottom = 40, 10, 10, 24
    plot_w = w - margin_left - margin_right
    plot_h = h - margin_top - margin_bottom

    canvas[:] = (25, 25, 25)
    cv2.rectangle(canvas, (margin_left, margin_top), (margin_left + plot_w, margin_top + plot_h), (60, 60, 60), 1)

    for i, val in enumerate([0, 64, 128, 192, 255]):
        y = margin_top + int(plot_h - (val / 255.0) * plot_h)
        color = (70, 70, 70) if i not in (0, 4) else (90, 90, 90)
        cv2.line(canvas, (margin_left, y), (margin_left + plot_w, y), color, 1)
        cv2.putText(canvas, str(val), (4, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    n = len(b_values)
    if n < 2:
        return

    def to_points(values: List[float]) -> np.ndarray:
        pts = []
        for i, v in enumerate(values):
            x = margin_left + int((i / (MAX_POINTS - 1)) * plot_w)
            y = margin_top + int(plot_h - (v / 255.0) * plot_h)
            pts.append((x, y))
        return np.array(pts, dtype=np.int32)

    cv2.polylines(canvas, [to_points(b_values)], False, (255, 80, 80), 2)
    cv2.polylines(canvas, [to_points(g_values)], False, (80, 255, 80), 2)
    cv2.polylines(canvas, [to_points(r_values)], False, (80, 80, 255), 2)

    var_max = max(1.0, VAR_Y_MAX_BASE * var_scale)

    def to_var_points(values: List[float]) -> np.ndarray:
        pts = []
        for i, v in enumerate(values):
            x = margin_left + int((i / (MAX_POINTS - 1)) * plot_w)
            y = margin_top + int(plot_h - (min(v, var_max) / var_max) * plot_h)
            pts.append((x, y))
        return np.array(pts, dtype=np.int32)

    if len(b_var_values) >= 2:
        cv2.polylines(canvas, [to_var_points(b_var_values)], False, (180, 50, 50), 1)
        cv2.polylines(canvas, [to_var_points(g_var_values)], False, (50, 180, 50), 1)
        cv2.polylines(canvas, [to_var_points(r_var_values)], False, (50, 50, 180), 1)

    cv2.putText(canvas, "Mean B/G/R", (w - 230, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Var b/g/r", (w - 110, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"samples: {n}/{MAX_POINTS} | var_scale: {var_scale:.2f} | +/- adjust", (margin_left, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


def fetch_jpeg_frame(session: requests.Session, url: str, timeout: float) -> Optional[np.ndarray]:
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        data = np.frombuffer(response.content, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return frame
    except requests.RequestException:
        return None


def main():
    global roi_rect

    parser = argparse.ArgumentParser(description="从URL循环获取JPEG并进行ROI颜色统计")
    parser.add_argument("url", help="JPEG图像URL，例如 http://127.0.0.1:8080/shot.jpg")
    parser.add_argument("--timeout", type=float, default=2.0, help="单次请求超时秒数")
    args = parser.parse_args()

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    b_hist: Deque[float] = collections.deque(maxlen=MAX_POINTS)
    g_hist: Deque[float] = collections.deque(maxlen=MAX_POINTS)
    r_hist: Deque[float] = collections.deque(maxlen=MAX_POINTS)
    b_var_hist: Deque[float] = collections.deque(maxlen=MAX_POINTS)
    g_var_hist: Deque[float] = collections.deque(maxlen=MAX_POINTS)
    r_var_hist: Deque[float] = collections.deque(maxlen=MAX_POINTS)
    var_scale = 1.0

    session = requests.Session()
    last_ts = time.time()
    fps = 0.0

    while True:
        frame = fetch_jpeg_frame(session, args.url, args.timeout)
        if frame is None:
            black = np.zeros((FRAME_HEIGHT + PLOT_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            cv2.putText(black, "Fetch failed, retrying...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, black)
            if (cv2.waitKey(30) & 0xFF) == ord("q"):
                break
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)
        draw_frame = frame.copy()

        if selecting and selection_start and selection_end:
            x1, y1, x2, y2 = normalize_rect(selection_start, selection_end)
            cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

        if roi_rect is not None:
            x1, y1, x2, y2 = roi_rect
            x1 = max(0, min(FRAME_WIDTH - 1, x1))
            x2 = max(0, min(FRAME_WIDTH - 1, x2))
            y1 = max(0, min(FRAME_HEIGHT - 1, y1))
            y2 = max(0, min(FRAME_HEIGHT - 1, y2))

            if x2 > x1 and y2 > y1:
                roi = frame[y1:y2, x1:x2]
                mean_bgr = cv2.mean(roi)[:3]
                b_var, g_var, r_var = np.var(roi.reshape(-1, 3), axis=0)
                b_hist.append(mean_bgr[0])
                g_hist.append(mean_bgr[1])
                r_hist.append(mean_bgr[2])
                b_var_hist.append(float(b_var))
                g_var_hist.append(float(g_var))
                r_var_hist.append(float(r_var))
                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(draw_frame, f"Mean B:{mean_bgr[0]:.1f} G:{mean_bgr[1]:.1f} R:{mean_bgr[2]:.1f}", (x1, max(40, y1 - 26)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(draw_frame, f"Var  B:{b_var:.1f} G:{g_var:.1f} R:{r_var:.1f}", (x1, max(24, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)

        plot_canvas = np.zeros((PLOT_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        draw_plot(plot_canvas, list(b_hist), list(g_hist), list(r_hist), list(b_var_hist), list(g_var_hist), list(r_var_hist), var_scale)
        combined = np.vstack((draw_frame, plot_canvas))

        now = time.time()
        dt = now - last_ts
        last_ts = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        cv2.putText(combined, f"FPS: {fps:.1f} | URL JPEG | q: quit | c: clear ROI", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            roi_rect = None
            b_hist.clear()
            g_hist.clear()
            r_hist.clear()
            b_var_hist.clear()
            g_var_hist.clear()
            r_var_hist.clear()
        if key in (ord("+"), ord("=")):
            var_scale = min(20.0, var_scale * 1.1)
        if key in (ord("-"), ord("_")):
            var_scale = max(0.05, var_scale / 1.1)

    session.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
