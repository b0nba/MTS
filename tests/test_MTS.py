import mts.mts as mts

import numpy as np

from mts.mts import MTS


def test_mts_machine_classification():

    # Columns:
    # 0 - temperature
    # 1 - pressure
    # 2 - vibration
    # 3 - current

    normal_data = np.array([
        [50.2, 100.5, 2.1, 10.0],
        [49.8,  99.7, 1.9, 10.4],
        [50.6, 101.2, 2.2,  9.8],
        [49.4, 100.8, 2.0, 10.1],
        [50.1,  98.9, 2.3,  9.7],
        [49.7, 101.0, 1.8, 10.3],
        [50.4,  99.5, 2.0,  9.9],
        [49.5, 100.2, 2.4, 10.2],
        [50.3, 100.7, 1.7,  9.6],
        [49.9,  99.1, 2.1, 10.5],
        [50.7, 100.0, 1.9, 10.0],
        [49.6, 101.3, 2.2,  9.8],
    ])

    abnormal_data = np.array([
        [56.0, 108.0, 4.5, 13.0],
        [44.0,  92.0, 0.7,  7.5],
        [55.0,  95.0, 4.0, 12.5],
        [45.0, 109.0, 0.9,  7.0],
        [58.0, 103.0, 5.2, 13.5],
        [43.0,  96.0, 0.6,  8.0],
        [54.0, 110.0, 4.8, 11.8],
        [46.0,  91.0, 1.0,  7.2],
    ])

    X_train = np.vstack([
        normal_data,
        abnormal_data
    ])

    # 1 = normal
    # 0 = abnormal
    y_train = np.array([
        1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1,

        0, 0, 0, 0,
        0, 0, 0, 0
    ])

    # L8-style orthogonal array
    # Four investigated variables.
    oa_design = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 0],
    ])

    model = MTS(
        opt=oa_design,
        alpha=0.05
    )

    model.fit(
        X_train,
        y_train
    )

    X_test = np.array([
        # Should look NORMAL
        [50.0, 100.1, 2.0, 10.1],
        [49.6,  99.8, 2.2,  9.9],

        # Should look ABNORMAL
        [56.0, 107.0, 4.3, 12.8],
        [44.5,  93.0, 0.8,  7.4],
    ])

    predictions = model.predict(X_test)

    expected = np.array([
        1,
        1,
        0,
        0
    ])

    np.testing.assert_array_equal(
        predictions,
        expected
    )