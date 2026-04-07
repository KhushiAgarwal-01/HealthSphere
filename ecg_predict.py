import os
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

try:
    from skimage import color, measure
    from skimage.filters import gaussian, threshold_otsu
    from skimage.io import imread
    from skimage.transform import resize
except Exception:
    color = None
    measure = None
    gaussian = None
    threshold_otsu = None
    imread = None
    resize = None


_MODELS_LOADED = False
_REPO_PCA = None
_REPO_CLASSIFIER = None


def _resolve_file(model_dir, candidates):
    for name in candidates:
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            return path
    return None


def _repo_divide_12_leads(gray_image):
    return [
        gray_image[300:600, 150:643],
        gray_image[300:600, 646:1135],
        gray_image[300:600, 1140:1625],
        gray_image[300:600, 1630:2125],
        gray_image[600:900, 150:643],
        gray_image[600:900, 646:1135],
        gray_image[600:900, 1140:1625],
        gray_image[600:900, 1630:2125],
        gray_image[900:1200, 150:643],
        gray_image[900:1200, 646:1135],
        gray_image[900:1200, 1140:1625],
        gray_image[900:1200, 1630:2125],
    ]


def _repo_extract_features(image_path):
    if any(
        x is None
        for x in [color, measure, gaussian, threshold_otsu, imread, resize]
    ):
        raise ImportError(
            "scikit-image is required for repo ECG pipeline. Please install scikit-image."
        )

    image = imread(image_path)
    image_gray = color.rgb2gray(image)
    image_gray = resize(image_gray, (1572, 2213))
    leads = _repo_divide_12_leads(image_gray)

    per_lead = []
    for lead in leads:
        blurred = gaussian(lead, sigma=0.7)
        global_thresh = threshold_otsu(blurred)
        binary = blurred < global_thresh
        binary = resize(binary, (300, 450))

        contours = measure.find_contours(binary, 0.8)
        if not contours:
            contour = np.zeros((255, 2), dtype=np.float32)
        else:
            contour = max(contours, key=lambda c: c.shape[0])
            contour = resize(contour, (255, 2))

        scaled = MinMaxScaler().fit_transform(contour)
        per_lead.append(scaled[:, 0])

    features = np.concatenate(per_lead, axis=0).reshape(1, -1)
    return features


def load_ecg_models():
    """Load only repo ECG models once at startup (no fallback)."""
    global _MODELS_LOADED
    global _REPO_PCA, _REPO_CLASSIFIER

    if _MODELS_LOADED:
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "saved_models", "ecg")

    repo_pca_path = _resolve_file(model_dir, ["PCA_ECG (1).pkl"])
    repo_model_path = _resolve_file(model_dir, ["Heart_Disease_Prediction_using_ECG (4).pkl"])

    if not repo_pca_path or not repo_model_path:
        raise FileNotFoundError(
            "Repo ECG model files are missing. Expected 'PCA_ECG (1).pkl' and "
            "'Heart_Disease_Prediction_using_ECG (4).pkl' in saved_models/ecg."
        )

    try:
        _REPO_PCA = joblib.load(repo_pca_path)
        _REPO_CLASSIFIER = joblib.load(repo_model_path)
    except Exception as e:
        raise RuntimeError(
            "Repo ECG model could not be loaded in this Python/sklearn environment. "
            "Use the same environment as the original repo deployment "
            "(older sklearn build) to run this model."
        ) from e

    _MODELS_LOADED = True


def _predict_repo(image_path):
    features = _repo_extract_features(image_path)
    reduced = _REPO_PCA.transform(features)
    pred = _REPO_CLASSIFIER.predict(reduced)
    cls = int(pred[0])
    mapping = {
        1: "You ECG corresponds to Myocardial Infarction",
        0: "You ECG corresponds to Abnormal Heartbeat",
        2: "Your ECG is Normal",
        3: "You ECG corresponds to History of Myocardial Infarction",
    }
    return mapping.get(cls, str(cls))


def predict_ecg(image_path):
    if not _MODELS_LOADED:
        load_ecg_models()
    return _predict_repo(image_path)
