import numpy as np

from mts.MTS import MTS


def test_mts_fit_and_predict():
    X_train = np.array([
        [1.0, 2.0, 3.0],
        [1.1, 2.2, 2.9],
        [0.9, 1.8, 3.1],
        [1.2, 2.1, 3.2],
        [0.8, 1.9, 2.8],

        [5.0, 6.0, 3.1],
        [4.8, 5.7, 3.0],
        [5.2, 6.2, 2.9],
    ])

    y_train = np.array([
        1, 1, 1, 1, 1,
        0, 0, 0
    ])

    oa_design = np.array([
        [1, 1, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    model = MTS(
        opt=oa_design,
        threshold=0.0
    )

    model.fit(X_train, y_train)

    X_test = np.array([
        [1.0, 2.1, 3.0],
        [5.5, 6.0, 3.1],
    ])

    predictions = model.predict(X_test)

    assert model.selected_features_.shape == (3,)
    assert predictions.shape == (2,)
    assert set(predictions).issubset({0, 1})