import numpy as np
from PIL import Image
import cv2
import io

def estimate_jpeg_quality(pil_image):
    # crude approximation: re-encode at various qualities and see size behavior
    buf = io.BytesIO()
    pil_image.save(buf, format='JPEG', quality=95)
    size95 = len(buf.getvalue())
    buf = io.BytesIO()
    pil_image.save(buf, format='JPEG', quality=50)
    size50 = len(buf.getvalue())
    # ratio indicates compression sensitivity
    return float(size95) / (float(size50) + 1e-9)

def laplacian_energy(pil_image):
    im = np.array(pil_image.convert('L'), dtype=np.float32)
    lap = cv2.Laplacian(im, cv2.CV_32F)
    return float(np.mean(np.abs(lap)))

def noise_level_estimate(pil_image):
    # estimate noise via median filter residual
    im = np.array(pil_image.convert('L'), dtype=np.float32)
    med = cv2.medianBlur(im.astype('uint8'), 3).astype(np.float32)
    residual = im - med
    return float(np.std(residual))

def image_forensic_score(pil_image):
    # composite score: higher values => more suspicious
    q = estimate_jpeg_quality(pil_image)
    lap = laplacian_energy(pil_image)
    noise = noise_level_estimate(pil_image)
    # normalize heuristically
    score = (lap * 0.4) + (noise * 0.4) + (1.0 / (q + 1e-6) * 0.2)
    return score
