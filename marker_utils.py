import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from debug_utils import save_step


def draw_marker_ids(img, marker_dict, title="Detected Markers with IDs"):
    img_vis = img.copy()

    for marker_id, data in marker_dict.items():
        corners = data['corners'].astype(int)
        center = tuple(np.mean(corners, axis=0).astype(int))

        cv.polylines(img_vis, [corners], True, (0, 255, 0), 2)
        cv.circle(img_vis, center, 6, (255, 0, 0), -1)
        cv.putText(img_vis, str(marker_id),
                   (center[0] - 15, center[1] + 15),
                   cv.FONT_HERSHEY_SIMPLEX,
                   1.2, (0, 255, 0), 3, cv.LINE_AA)

    plt.figure(figsize=(10, 14))
    plt.imshow(cv.cvtColor(img_vis, cv.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

    save_step(img_vis, "Detected_Markers_with_IDs")


def findArucoMarkers(imgGray):
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_APRILTAG_16H5)
    aruco_params = cv.aruco.DetectorParameters()

    detector = cv.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, rejected = detector.detectMarkers(imgGray)
    print(f"Tìm thấy {len(corners) if corners is not None else 0} ArUco markers")

    if not corners or len(corners) < 4:
        raise RuntimeError(f"Không tìm thấy đủ markers. Chỉ tìm thấy {len(corners) if corners else 0}")

    marker_dict = {}
    for i, corner in enumerate(corners):
        center = np.mean(corner[0], axis=0)
        marker_id = ids[i][0] if ids is not None else i
        marker_dict[marker_id] = {
            'center': center,
            'corners': corner[0]
        }
        print(f"[MARKER] ID: {marker_id}, Center: ({center[0]:.1f}, {center[1]:.1f})")

    return marker_dict


def getMainCorners(marker_dict):
    required_ids = [1, 3, 2, 4]

    if not all(id in marker_dict for id in required_ids):
        missing = [id for id in required_ids if id not in marker_dict]
        raise RuntimeError(f"Không tìm thấy đủ marker góc chính. Thiếu: {missing}")

    tl = marker_dict[1]['center']
    tr = marker_dict[3]['center']
    bl = marker_dict[2]['center']
    br = marker_dict[4]['center']

    print(f"[MAIN CORNERS] TL(ID1): {tl}, TR(ID3): {tr}, BR(ID4): {br}, BL(ID2): {bl}")

    return np.array([tl, tr, br, bl], dtype="float32")