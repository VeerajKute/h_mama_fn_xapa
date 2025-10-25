from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import pytesseract
import torch
import os
from hmama.config_small import ModelConfigSmall
from hmama.models.text_encoder_full import PretrainedTextEncoder
from hmama.models.clip_encoder_ft import CLIPEncoder
from hmama.models.ug_mop_adaptive import UGMOP_Adaptive
from hmama.models.fusion import CoAttentionFusion
from hmama.models.aigc_sentinel import TextAIGCSentinel, ImageArtifactSentinel
from hmama.explain.explain_utils import text_integrated_gradients, image_gradient_saliency
from hmama.models.adaptive_calibration import AdaptiveCalibration
from hmama.models.dynamic_confidence import DynamicConfidenceAdjuster


app = FastAPI()
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
cfg = ModelConfigSmall(device=device)


class PredictRequest(BaseModel):
    text: str
    image_path: Optional[str] = None

class FeedbackRequest(BaseModel):
    text: str
    predicted_label: str
    predicted_confidence: float
    actual_label: str
    user_id: Optional[str] = None


# Initialize encoders (lite defaults)
text_enc = PretrainedTextEncoder(model_name=cfg.text_model_name, whiten_dim=cfg.whiten_dim, device=cfg.device)
clip_enc = CLIPEncoder(model_name=cfg.clip_model_name, device=cfg.device)
ug = UGMOP_Adaptive(in_text=text_enc.whiten_dim, in_ocr=text_enc.whiten_dim, in_img=512, device=cfg.device)
fusion = CoAttentionFusion(dim_text=text_enc.whiten_dim, dim_img=512)

# load checkpoint if exists
try:
    ck = torch.load('checkpoints/model_full.pth', map_location=device)
    if 'ug' in ck:
        ug.load_state_dict(ck['ug'])
    if 'fusion' in ck:
        fusion.load_state_dict(ck['fusion'])
    print('Loaded checkpoint.')
except Exception as e:
    print('No checkpoint found or failed to load:', e)

t_sentinel = TextAIGCSentinel(device=device)
i_sentinel = ImageArtifactSentinel()

# Initialize adaptive systems
calibration_system = AdaptiveCalibration()
confidence_adjuster = DynamicConfidenceAdjuster()


@app.post('/predict')
def predict(req: PredictRequest):
    # Handle case where no image is provided
    if req.image_path is None:
        # Use text-only analysis
        zt = text_enc.encode([req.text]).to(device)
        zo = text_enc.encode(['']).to(device)  # Empty OCR
        # Create dummy image encoding (zeros)
        zi = torch.zeros(1, 512).to(device)
        contr = torch.tensor([0.5]).to(device)  # Neutral contradiction score
        # AIGC
        flag_text, _ = t_sentinel.suspicious(req.text)
        flag_img = 0.0  # No image
        combined = min(1.0, flag_text + flag_img)
    else:
        if not os.path.exists(req.image_path):
            raise HTTPException(status_code=400, detail=f"Image not found: {req.image_path}")
        try:
            img = Image.open(req.image_path).convert('RGB')
        except Exception:
            raise HTTPException(status_code=400, detail="Failed to open image. Ensure a valid image file path.")
        # OCR
        try:
            ocr_text = pytesseract.image_to_string(img)[:400]
        except Exception:
            ocr_text = ''
        # encode
        zt = text_enc.encode([req.text]).to(device)
        zo = text_enc.encode([ocr_text]).to(device)
        zi = clip_enc.encode_image([img]).to(device)
        contr = clip_enc.contradiction_score([req.text], [img]).to(device)
        # AIGC
        flag_text, _ = t_sentinel.suspicious(req.text)
        flag_img, _ = i_sentinel.suspicious(img)
        combined = min(1.0, flag_text + flag_img)
    
    fused_logits, gate_w = ug(zt, zo, zi, contr, aigc_flag=torch.tensor([combined], device=device))
    # final probability
    probs = torch.softmax(fused_logits, dim=-1).detach().cpu().numpy().tolist()[0]
    raw_confidence = max(probs)
    label = 'FAKE' if probs[0] >= probs[1] else 'REAL'
    
    # Apply dynamic confidence adjustment
    has_image = req.image_path is not None
    adjusted_confidence, adjustment_factors = confidence_adjuster.adjust_confidence(
        req.text, raw_confidence, label, has_image
    )
    
    # Apply calibration system adjustments
    calibrated_confidence = calibration_system.get_adjusted_confidence(adjusted_confidence, label)
    
    return {
        'label': label,
        'confidence': round(float(calibrated_confidence) * 100.0, 2),
        'raw_confidence': round(float(raw_confidence) * 100.0, 2),
        'confidence_unit': 'percent',
        'probabilities': {'fake': round(float(probs[0]) * 100.0, 2), 'real': round(float(probs[1]) * 100.0, 2)},
        'gate_weights': gate_w.detach().cpu().tolist(),
        'adjustment_factors': adjustment_factors,
        'uncertainty_indicators': confidence_adjuster.get_uncertainty_indicators(req.text)
    }


@app.post('/explain')
def explain(req: PredictRequest):
    # Handle case where no image is provided
    if req.image_path is None:
        # Use text-only analysis
        img = None
        ocr_text = ''
        # Get model encodings and predictions
        zt = text_enc.encode([req.text]).to(device)
        zo = text_enc.encode([ocr_text]).to(device)
        zi = torch.zeros(1, 512).to(device)  # Dummy image encoding
        contr = torch.tensor([0.5]).to(device)  # Neutral contradiction score
        
        # AIGC detection
        flag_text, _ = t_sentinel.suspicious(req.text)
        flag_img = 0.0  # No image
        combined = min(1.0, flag_text + flag_img)
    else:
        if not os.path.exists(req.image_path):
            raise HTTPException(status_code=400, detail=f"Image not found: {req.image_path}")
        try:
            img = Image.open(req.image_path).convert('RGB')
        except Exception:
            raise HTTPException(status_code=400, detail="Failed to open image. Ensure a valid image file path.")
        
        # Run the same forward pass as /predict to get all model outputs
        try:
            ocr_text = pytesseract.image_to_string(img)[:400]
        except Exception:
            ocr_text = ''
        
        # Get model encodings and predictions
        zt = text_enc.encode([req.text]).to(device)
        zo = text_enc.encode([ocr_text]).to(device)
        zi = clip_enc.encode_image([img]).to(device)
        contr = clip_enc.contradiction_score([req.text], [img]).to(device)
        
        # AIGC detection
        flag_text, _ = t_sentinel.suspicious(req.text)
        flag_img, _ = i_sentinel.suspicious(img)
        combined = min(1.0, flag_text + flag_img)
    
    # Get final prediction
    fused_logits, gate_w = ug(zt, zo, zi, contr, aigc_flag=torch.tensor([combined], device=device))
    probs = torch.softmax(fused_logits, dim=-1).detach().cpu().numpy().tolist()[0]
    label = 'FAKE' if probs[0] >= probs[1] else 'REAL'
    confidence = max(probs)
    
    # Get contradiction score for image-text alignment
    contradiction_score = contr.detach().cpu().item()
    
    # Text attributions for human-readable explanation
    toks = []
    try:
        import torch.nn as nn
        from captum.attr import IntegratedGradients
        enc = text_enc.tokenizer(req.text, return_tensors='pt', truncation=True, max_length=160)
        input_ids = enc['input_ids'].to(device)
        att = enc['attention_mask'].to(device)
        with torch.no_grad():
            out = text_enc.model(input_ids=input_ids, attention_mask=att)
            cls = out.last_hidden_state[:,0]  # (1, hidden)
        probe = nn.Linear(cls.size(-1), 2).to(device)
        # forward for IG: recompute CLS inside to keep gradients
        def fwd(ids):
            o = text_enc.model(input_ids=ids, attention_mask=att)
            c = o.last_hidden_state[:,0]
            return probe(c)
        ig = IntegratedGradients(fwd)
        attributions, _ = ig.attribute(inputs=input_ids, target=1, return_convergence_delta=True)
        at = attributions.sum(dim=-1).squeeze(0)
        tokens = text_enc.tokenizer.convert_ids_to_tokens(input_ids[0])
        s = at.detach().cpu().numpy()
        s = (s - s.min()) / (s.max() - s.min() + 1e-12)
        toks = list(zip(tokens, s.tolist()))
    except Exception:
        try:
            # fallback to simple splitter
            from hmama.explain.simple import explain_text_simple
            toks = explain_text_simple(req.text)
        except Exception:
            toks = []
    
    # Generate human-readable explanations
    def _clean_token(tok: str) -> str:
        if tok.startswith('##'):
            return tok[2:]
        return tok
    
    def _generate_text_explanation(tokens_with_scores, prediction_label):
        """Generate human-readable text explanation based on token attributions."""
        try:
            # Get top influential words (excluding special tokens)
            candidates = [(t, s) for (t, s) in tokens_with_scores if t not in ['[CLS]', '[SEP]', '[PAD]']]
            candidates = [(_clean_token(t), s) for (t, s) in candidates if _clean_token(t).isalpha() and len(_clean_token(t)) > 2]
            
            if not candidates:
                return None
                
            # Sort by influence score and get top 3
            top_words = sorted(candidates, key=lambda x: -x[1])[:3]
            word_names = [w for w, _ in top_words]
            
            if prediction_label == 'FAKE':
                return f"The model predicts FAKE because words like '{word_names[0]}', '{word_names[1]}' strongly influenced the decision."
            else:
                return f"The model predicts REAL with supporting words like '{word_names[0]}', '{word_names[1]}'."
        except Exception:
            return None
    
    def _generate_image_explanation(contradiction_score, prediction_label, has_image):
        """Generate human-readable image explanation based on text-image alignment."""
        try:
            if not has_image:
                return "No image provided for analysis."
            elif contradiction_score > 0.7:
                return "The image contradicts the text content."
            elif contradiction_score < 0.3:
                return "The image supports the text content."
            else:
                return "The image is somewhat related to the text but not strongly aligned."
        except Exception:
            return None
    
    def _generate_aigc_explanation(text_flag, img_flag):
        """Generate explanation for AI-generated content detection."""
        explanations = []
        if text_flag > 0.5:
            explanations.append("The text appears to be AI-generated.")
        if img_flag > 0.5:
            explanations.append("The image appears manipulated or AI-generated.")
        return explanations
    
    def _generate_counterfactual_explanation(tokens_with_scores):
        """Generate counterfactual reasoning based on most influential token."""
        try:
            candidates = [(t, s) for (t, s) in tokens_with_scores if t not in ['[CLS]', '[SEP]', '[PAD]']]
            candidates = [(_clean_token(t), s) for (t, s) in candidates if _clean_token(t).isalpha() and len(_clean_token(t)) > 2]
            
            if not candidates:
                return None
                
            top_token, top_score = max(candidates, key=lambda x: x[1])
            if top_score > 0.5:  # Only mention if highly influential
                return f"If the word '{top_token}' was removed, the prediction might change."
        except Exception:
            return None
    
    # Generate all explanation components
    text_explanation = _generate_text_explanation(toks, label)
    image_explanation = _generate_image_explanation(contradiction_score, label, img is not None)
    aigc_explanations = _generate_aigc_explanation(flag_text, flag_img)
    counterfactual_explanation = _generate_counterfactual_explanation(toks)
    
    # Combine into coherent paragraph
    explanation_parts = []
    
    if text_explanation:
        explanation_parts.append(text_explanation)
    
    if image_explanation:
        explanation_parts.append(image_explanation)
    
    if aigc_explanations:
        explanation_parts.extend(aigc_explanations)
    
    if counterfactual_explanation:
        explanation_parts.append(counterfactual_explanation)
    
    # Create final human-readable explanation
    human_explanation = " ".join(explanation_parts) if explanation_parts else "The model made this prediction based on the overall content analysis."
    
    # Prepare counterfactual info for optional return
    counterfactual_info = None
    if counterfactual_explanation:
        try:
            candidates = [(t, s) for (t, s) in toks if t not in ['[CLS]', '[SEP]', '[PAD]']]
            candidates = [(_clean_token(t), s) for (t, s) in candidates if _clean_token(t).isalpha() and len(_clean_token(t)) > 2]
            if candidates:
                top_token, _ = max(candidates, key=lambda x: x[1])
                counterfactual_info = {'removed_token': top_token, 'explanation': counterfactual_explanation}
        except Exception:
            pass
    
    # Apply dynamic confidence adjustment
    has_image = req.image_path is not None
    adjusted_confidence, adjustment_factors = confidence_adjuster.adjust_confidence(
        req.text, confidence, label, has_image
    )
    
    # Apply calibration system adjustments
    calibrated_confidence = calibration_system.get_adjusted_confidence(adjusted_confidence, label)
    
    # Get model health metrics
    health_metrics = calibration_system.get_uncertainty_metrics()
    
    return {
        'label': label,
        'confidence': round(float(calibrated_confidence) * 100.0, 2),
        'raw_confidence': round(float(confidence) * 100.0, 2),
        'confidence_unit': 'percent',
        'human_explanation': human_explanation,
        'probabilities': {'fake': round(float(probs[0]) * 100.0, 2), 'real': round(float(probs[1]) * 100.0, 2)},
        'gate_weights': gate_w.detach().cpu().tolist(),
        'text_attributions': toks,
        'contradiction_score': round(contradiction_score, 3),
        'aigc_flags': {'text': round(float(flag_text), 3), 'image': round(float(flag_img), 3)},
        'counterfactual': counterfactual_info,
        'adjustment_factors': adjustment_factors,
        'uncertainty_indicators': confidence_adjuster.get_uncertainty_indicators(req.text),
        'model_health': health_metrics,
        'confidence_explanation': confidence_adjuster.get_confidence_explanation(
            req.text, confidence, calibrated_confidence, adjustment_factors
        )
    }


@app.post('/feedback')
def feedback(req: FeedbackRequest):
    """Submit feedback to improve model accuracy"""
    try:
        # Add feedback to calibration system
        calibration_system.add_feedback(
            text=req.text,
            predicted_label=req.predicted_label,
            predicted_confidence=req.predicted_confidence,
            actual_label=req.actual_label,
            user_id=req.user_id
        )
        
        # Get updated model health
        health_metrics = calibration_system.get_uncertainty_metrics()
        recommendations = calibration_system.get_recommendations()
        
        return {
            'status': 'success',
            'message': 'Feedback received and processed',
            'model_health': health_metrics,
            'recommendations': recommendations
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error processing feedback: {str(e)}'
        }

@app.get('/model_health')
def model_health():
    """Get current model health and performance metrics"""
    try:
        health_metrics = calibration_system.get_uncertainty_metrics()
        recommendations = calibration_system.get_recommendations()
        
        return {
            'status': 'success',
            'health_metrics': health_metrics,
            'recommendations': recommendations,
            'feedback_count': len(calibration_system.feedback_history)
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error getting model health: {str(e)}'
        }

@app.get('/health')
def health():
    ok = True
    details = {}
    try:
        _ = text_enc is not None and clip_enc is not None and ug is not None
        details['models'] = 'loaded'
    except Exception as e:
        ok = False
        details['models'] = f'error: {e}'
    return {'status': 'ok' if ok else 'error', 'details': details}
