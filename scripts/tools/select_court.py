"""Click TL -> TR -> BR -> BL on the first video frame and print court_points string."""
import os
import sys

import cv2


def main():
    if len(sys.argv) < 2:
        print("usage: python3 _select_court.py <video_or_image_path>")
        sys.exit(1)

    src = sys.argv[1]
    if not os.path.exists(src):
        print(f"not found: {src}")
        sys.exit(1)

    if src.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        frame = cv2.imread(src)
    else:
        cap = cv2.VideoCapture(src)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print("cannot read frame")
            sys.exit(1)

    h, w = frame.shape[:2]
    print(f"frame size: {w}x{h}")
    print("Click 4 corners IN ORDER:  TL -> TR -> BR -> BL")
    print("Press 'r' to reset, 'q' or ESC when done with 4 points.")

    pts = []
    win = "Select 4 court corners (TL TR BR BL)"

    def on_mouse(event, x, y, flags, param):
        nonlocal pts
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
            print(f"  point {len(pts)}: ({x}, {y})")

    labels = ["TL", "TR", "BR", "BL"]
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(w * 2, 1600), min(h * 2, 900))
    cv2.setMouseCallback(win, on_mouse)

    while True:
        canvas = frame.copy()
        for i, p in enumerate(pts):
            cv2.circle(canvas, p, 6, (0, 255, 255), -1)
            cv2.putText(canvas, labels[i], (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        if len(pts) >= 2:
            for i in range(len(pts)):
                j = (i + 1) % len(pts)
                if j < len(pts):
                    cv2.line(canvas, pts[i], pts[j], (0, 255, 0), 2, cv2.LINE_AA)
            if len(pts) == 4:
                cv2.line(canvas, pts[3], pts[0], (0, 255, 0), 2, cv2.LINE_AA)
        tip = f"clicked {len(pts)}/4 - next: {labels[len(pts)] if len(pts)<4 else 'done, press q'}"
        cv2.putText(canvas, tip, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(win, canvas)
        k = cv2.waitKey(20) & 0xFF
        if k in (ord('q'), 27) and len(pts) == 4:
            break
        if k == ord('r'):
            pts = []
            print("reset")

    cv2.destroyAllWindows()
    s = ",".join(f"{x},{y}" for (x, y) in pts)
    print()
    print("====== copy this ======")
    print(f'--court_points "{s}"')
    print("=======================")


if __name__ == "__main__":
    main()
