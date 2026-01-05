import torch
import numpy as np
from typing import Tuple, Dict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_ece(
    probabilities: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Expected Calibration Error (ECE)
    
    Args:
        probabilities: [N, num_classes] predicted probabilities
        predictions: [N] predicted classes
        labels: [N] true labels
        num_bins: Number of bins for calibration
        
    Returns:
        ece: Expected Calibration Error
        bin_accuracies: Accuracy per bin
        bin_confidences: Average confidence per bin
        bin_counts: Number of samples per bin
    """
    # Get confidence (max probability)
    confidences = np.max(probabilities, axis=1)
    
    # Check if predictions are correct
    accuracies = (predictions == labels).astype(float)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_accuracies = np.zeros(num_bins)
    bin_confidences = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)
    
    ece = 0.0
    
    for i in range(num_bins):
        # Find samples in this bin
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_count = np.sum(in_bin)
        
        if bin_count > 0:
            # Average accuracy in bin
            bin_accuracy = np.mean(accuracies[in_bin])
            # Average confidence in bin
            bin_confidence = np.mean(confidences[in_bin])
            
            # ECE contribution
            ece += (bin_count / len(confidences)) * np.abs(bin_accuracy - bin_confidence)
            
            # Store for plotting
            bin_accuracies[i] = bin_accuracy
            bin_confidences[i] = bin_confidence
            bin_counts[i] = bin_count
    
    return ece, bin_accuracies, bin_confidences, bin_counts


def compute_brier_score(
    probabilities: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Compute Brier Score
    
    Args:
        probabilities: [N, num_classes] predicted probabilities
        labels: [N] true labels
        
    Returns:
        brier_score: Mean squared error of probabilities
    """
    num_samples = len(labels)
    num_classes = probabilities.shape[1]
    
    # One-hot encode labels
    labels_one_hot = np.zeros((num_samples, num_classes))
    labels_one_hot[np.arange(num_samples), labels] = 1
    
    # Brier score: mean squared error
    brier = np.mean(np.sum((probabilities - labels_one_hot) ** 2, axis=1))
    
    return brier


def compute_nll(
    probabilities: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Compute Negative Log-Likelihood (NLL)
    
    Args:
        probabilities: [N, num_classes] predicted probabilities
        labels: [N] true labels
        
    Returns:
        nll: Negative log-likelihood
    """
    num_samples = len(labels)
    
    # Get probability of true class
    true_class_probs = probabilities[np.arange(num_samples), labels]
    
    # Add epsilon for numerical stability
    true_class_probs = np.clip(true_class_probs, 1e-10, 1.0)
    
    # NLL
    nll = -np.mean(np.log(true_class_probs))
    
    return nll


def compute_aurc(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    num_points: int = 100
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Area Under Risk-Coverage Curve (AURC)
    
    Lower is better. AURC measures the area under the curve of
    error rate vs coverage.
    
    Args:
        confidences: [N] confidence scores
        predictions: [N] predicted classes
        labels: [N] true labels
        num_points: Number of points in curve
        
    Returns:
        aurc: Area under risk-coverage curve
        coverage_points: Coverage values
        risk_points: Error rates at each coverage
    """
    N = len(confidences)
    
    # Sort by confidence (descending)
    sorted_indices = np.argsort(-confidences)
    sorted_predictions = predictions[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    coverage_points = []
    risk_points = []
    
    for i in range(1, num_points + 1):
        # Coverage fraction
        coverage = i / num_points
        num_samples = max(1, int(N * coverage))
        
        # Select top confident samples
        selected_preds = sorted_predictions[:num_samples]
        selected_labels = sorted_labels[:num_samples]
        
        # Compute error rate (risk)
        errors = (selected_preds != selected_labels).sum()
        risk = errors / num_samples
        
        coverage_points.append(coverage)
        risk_points.append(risk)
    
    coverage_points = np.array(coverage_points)
    risk_points = np.array(risk_points)
    
    # Compute area under curve using trapezoidal rule
    aurc = np.trapz(risk_points, coverage_points)
    
    return aurc, coverage_points, risk_points


def compute_selective_accuracy(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    coverage_levels: list = [0.7, 0.8, 0.9, 1.0]
) -> Dict[float, float]:
    """
    Compute accuracy at different coverage levels
    
    Args:
        confidences: [N] confidence scores
        predictions: [N] predicted classes
        labels: [N] true labels
        coverage_levels: List of coverage thresholds
        
    Returns:
        Dictionary mapping coverage → accuracy
    """
    N = len(confidences)
    
    # Sort by confidence
    sorted_indices = np.argsort(-confidences)
    sorted_predictions = predictions[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    results = {}
    for coverage in coverage_levels:
        num_samples = int(N * coverage)
        if num_samples == 0:
            continue
        
        selected_preds = sorted_predictions[:num_samples]
        selected_labels = sorted_labels[:num_samples]
        
        accuracy = (selected_preds == selected_labels).mean()
        results[coverage] = accuracy
    
    return results


def compute_all_metrics(
    probabilities: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    confidences: np.ndarray,
    num_bins: int = 15
) -> Dict[str, float]:
    """
    Compute all calibration metrics
    
    Args:
        probabilities: [N, num_classes] predicted probabilities
        predictions: [N] predicted classes
        labels: [N] true labels
        confidences: [N] confidence scores
        num_bins: Number of bins for ECE
        
    Returns:
        Dictionary with all metrics
    """
    # ECE
    ece, _, _, _ = compute_ece(probabilities, predictions, labels, num_bins)
    
    # Brier Score
    brier = compute_brier_score(probabilities, labels)
    
    # NLL
    nll = compute_nll(probabilities, labels)
    
    # AURC
    aurc, _, _ = compute_aurc(confidences, predictions, labels)
    
    # Accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # F1 Score
    if len(np.unique(labels)) == 2:
        f1 = f1_score(labels, predictions, average='binary')
    else:
        f1 = f1_score(labels, predictions, average='macro')
    
    # Selective Accuracy
    selective_acc = compute_selective_accuracy(
        confidences, predictions, labels,
        coverage_levels=[0.7, 0.8, 0.9, 1.0]
    )
    
    return {
        'ece': ece,
        'brier': brier,
        'nll': nll,
        'aurc': aurc,
        'accuracy': accuracy,
        'f1': f1,
        'selective_acc_70': selective_acc.get(0.7, 0.0),
        'selective_acc_80': selective_acc.get(0.8, 0.0),
        'selective_acc_90': selective_acc.get(0.9, 0.0),
        'selective_acc_100': selective_acc.get(1.0, 0.0),
    }


def test_metrics():
    """Test calibration metrics"""
    print("Testing Calibration Metrics...")
    
    # Dummy data
    N = 1000
    num_classes = 3
    
    np.random.seed(42)
    probabilities = np.random.dirichlet(np.ones(num_classes), size=N)
    predictions = np.argmax(probabilities, axis=1)
    labels = np.random.randint(0, num_classes, size=N)
    confidences = np.max(probabilities, axis=1)
    
    # Compute metrics
    metrics = compute_all_metrics(probabilities, predictions, labels, confidences)
    
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✓ All metrics computed successfully!")


if __name__ == "__main__":
    test_metrics()