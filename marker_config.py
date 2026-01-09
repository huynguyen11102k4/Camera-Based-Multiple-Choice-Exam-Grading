import numpy as np

MARKER_CENTERS = {
    7:  (1265.0, 125.5),
    9:  (2111.8, 125.5),
    13: (438.5, 1055.2),
    14: (439.0, 2161.0),
    15: (1264.8, 1054.8),
    18: (2109.5, 2161.0),
    16: (1264.2, 2161.2),
    17: (2110.2, 1055.2),
    19: (438.2, 3306.8),
    20: (1263.8, 3307.2),
    21: (2111.8, 3306.8),
}

REGION_CONFIG = {
    'student_id': {
        'marker_ids': [7, 9, 17, 15],
        'dst_corners': np.array([
            MARKER_CENTERS[7],
            MARKER_CENTERS[9],
            MARKER_CENTERS[17],
            MARKER_CENTERS[15]
        ], dtype="float32")
    },
    'quiz_id': {
        'marker_ids': [7, 9, 17, 15],
        'dst_corners': np.array([
            MARKER_CENTERS[7],
            MARKER_CENTERS[9],
            MARKER_CENTERS[17],
            MARKER_CENTERS[15]
        ], dtype="float32")
    },
    'questions_1_10': {
        'marker_ids': [13, 15, 16, 14],
        'dst_corners': np.array([
            MARKER_CENTERS[13],
            MARKER_CENTERS[15],
            MARKER_CENTERS[16],
            MARKER_CENTERS[14]
        ], dtype="float32")
    },
    'questions_11_20': {
        'marker_ids': [15, 17, 18, 16],
        'dst_corners': np.array([
            MARKER_CENTERS[15],
            MARKER_CENTERS[17],
            MARKER_CENTERS[18],
            MARKER_CENTERS[16]
        ], dtype="float32")
    },
    'questions_21_30': {
        'marker_ids': [14, 16, 20, 19],
        'dst_corners': np.array([
            MARKER_CENTERS[14],
            MARKER_CENTERS[16],
            MARKER_CENTERS[20],
            MARKER_CENTERS[19]
        ], dtype="float32")
    },
    'questions_31_40': {
        'marker_ids': [16, 18, 21, 20],
        'dst_corners': np.array([
            MARKER_CENTERS[16],
            MARKER_CENTERS[18],
            MARKER_CENTERS[21],
            MARKER_CENTERS[20]
        ], dtype="float32")
    },
    'id_area': {
        'marker_ids': [7, 9, 17, 15],
        'dst_corners': np.array([
            MARKER_CENTERS[7],
            MARKER_CENTERS[9],
            MARKER_CENTERS[17],
            MARKER_CENTERS[15]
        ], dtype="float32")
    },
    'questions_full': {
        'marker_ids': [13, 17, 21, 19],
        'dst_corners': np.array([
            MARKER_CENTERS[13],
            MARKER_CENTERS[17],
            MARKER_CENTERS[21],
            MARKER_CENTERS[19]
        ], dtype="float32")
    }
}
