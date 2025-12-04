import os
import sys

# add project root (…/project) to Python path
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.predict import predict


def test_predict_output():
    # Example Iris features: sepal_length, sepal_width, petal_length, petal_width
    features = [5.1, 3.5, 1.4, 0.2]

    class_idx, name, proba = predict(features)

    # class index should be 0, 1 or 2
    assert isinstance(class_idx, int)
    assert 0 <= class_idx <= 2

    # class name should be a non-empty string
    assert isinstance(name, str)
    assert len(name) > 0

    # probabilities should be a length-3 distribution
    assert len(proba) == 3
    assert abs(float(proba.sum()) - 1.0) < 1e-6
