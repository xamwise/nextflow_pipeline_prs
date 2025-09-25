import numpy as np
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_nagelkerke_r2(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Calculate Nagelkerke's R² (pseudo-R²) for binary classification with robust error handling.
    
    Args:
        predictions: Model logits/predictions 
        targets: Binary targets (0/1)
        
    Returns:
        Nagelkerke's R² value
    """
    try:
        # Flatten arrays and convert to float
        predictions_flat = predictions.flatten().astype(float)
        targets_flat = targets.flatten().astype(float)
        
        # Check for invalid inputs
        if len(predictions_flat) != len(targets_flat):
            logger.warning("Predictions and targets have different lengths")
            return 0.0
        
        if len(predictions_flat) == 0:
            logger.warning("Empty predictions array")
            return 0.0
        
        # Remove any NaN or infinite values
        valid_mask = np.isfinite(predictions_flat) & np.isfinite(targets_flat)
        if not np.any(valid_mask):
            logger.warning("No valid (finite) values found")
            return 0.0
        
        predictions_flat = predictions_flat[valid_mask]
        targets_flat = targets_flat[valid_mask]
        
        # Convert logits to probabilities if needed (more robust sigmoid)
        if predictions_flat.min() < 0 or predictions_flat.max() > 1:
            # Clip extreme logits to prevent overflow in exp
            predictions_flat = np.clip(predictions_flat, -500, 500)
            probs = 1 / (1 + np.exp(-predictions_flat))
        else:
            probs = predictions_flat.copy()
        
        # Ensure targets are binary
        targets_flat = targets_flat.astype(float)
        if not np.all(np.isin(targets_flat, [0, 1])):
            logger.warning("Targets contain non-binary values")
            return 0.0
        
        # Check if we have both classes
        if len(np.unique(targets_flat)) < 2:
            logger.warning("Only one class present in targets")
            return 0.0
        
        n = len(targets_flat)
        
        # More aggressive clipping to avoid log(0)
        # Use epsilon that's much larger than machine precision
        epsilon = 1e-8
        probs = np.clip(probs, epsilon, 1 - epsilon)
        
        # Calculate null model probability with same clipping
        p_null = np.mean(targets_flat)
        p_null = np.clip(p_null, epsilon, 1 - epsilon)
        
        # Log-likelihood for null model (intercept-only)
        ll_null = np.sum(
            targets_flat * np.log(p_null) + (1 - targets_flat) * np.log(1 - p_null)
        )
        
        # Log-likelihood for fitted model
        ll_model = np.sum(
            targets_flat * np.log(probs) + (1 - targets_flat) * np.log(1 - probs)
        )
        
        # Check for invalid log-likelihoods
        if not np.isfinite(ll_null) or not np.isfinite(ll_model):
            logger.warning("Invalid log-likelihood values computed")
            return 0.0
        
        # Cox-Snell R²
        if ll_null >= ll_model:  # Model should not be worse than null
            logger.warning("Model performs worse than null model")
            return 0.0
        
        r2_cox_snell = 1 - np.exp((2/n) * (ll_null - ll_model))
        
        # Maximum possible Cox-Snell R² 
        r2_max = 1 - np.exp((2/n) * ll_null)
        
        # Nagelkerke's R² (normalized Cox-Snell)
        if r2_max > 0:
            r2_nagelkerke = r2_cox_snell / r2_max
        else:
            logger.warning("Maximum R² is zero or negative")
            return 0.0
        
        # Final validation and clipping
        r2_nagelkerke = float(np.clip(r2_nagelkerke, 0.0, 1.0))
        
        # Sanity check
        if not np.isfinite(r2_nagelkerke):
            logger.warning("Computed Nagelkerke R² is not finite")
            return 0.0
        
        return r2_nagelkerke
        
    except Exception as e:
        logger.error(f"Error calculating Nagelkerke R²: {str(e)}")
        return 0.0