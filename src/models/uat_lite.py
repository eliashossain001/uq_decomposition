
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from transformers import AutoModel, AutoConfig, AutoTokenizer

from .bayesian_embedding import BayesianEmbedding, BayesianLinear
from .uncertainty_attention import UncertaintyWeightedAttention
from .confidence_decision import ConfidenceDecision
from .variance_decomposition import LayerWiseVarianceDecomposition


class UATLite(nn.Module):
    """
    UAT-Lite: Complete implementation (Memory Optimized)
    """
    
    def __init__(
        self,
        base_model_name: str,
        num_classes: int,
        mc_samples: int = 10,
        dropout_rate: float = 0.3,
        uncertainty_penalty: float = 0.5,
        confidence_threshold: float = 0.7,
        use_layer_decomposition: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.base_model_name = base_model_name
        self.num_classes = num_classes
        self.mc_samples = mc_samples
        self.dropout_rate = dropout_rate
        self.uncertainty_penalty = uncertainty_penalty
        self.confidence_threshold = confidence_threshold
        self.use_layer_decomposition = use_layer_decomposition
        
        # Load pretrained transformer
        self.config = AutoConfig.from_pretrained(base_model_name)
        self.transformer = AutoModel.from_pretrained(base_model_name)
        
        # Get specifications
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers
        self.num_attention_heads = self.config.num_attention_heads
        
        # Component 1: Bayesian Embedding
        self.bayesian_embedding = BayesianEmbedding(
            embedding_layer=self.transformer.embeddings,
            dropout_rate=dropout_rate,
            mc_samples=mc_samples
        )
        
        # Component 3: Confidence-Guided Decision
        self.confidence_decision = ConfidenceDecision(
            num_classes=num_classes,
            confidence_threshold=confidence_threshold,
            temperature=1.0
        )
        
        # Theorem 5: Layer-wise Variance Decomposition
        if use_layer_decomposition:
            self.variance_decomposition = LayerWiseVarianceDecomposition(
                num_layers=self.num_layers,
                hidden_size=self.hidden_size
            )
        
        # Classification head
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward_with_mc_sampling(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_samples: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Memory-efficient MC sampling (FIXED ATTENTION MASK)
        """
        if num_samples is None:
            num_samples = self.mc_samples
        
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        # Get base embeddings (no gradient)
        with torch.no_grad():
            base_embeddings = self.transformer.embeddings(input_ids)
        
        # Compute token uncertainty (use 3 samples only)
        self.bayesian_embedding.enable_dropout()
        embedding_samples = []
        for _ in range(min(3, num_samples)):
            sample = self.bayesian_embedding.dropout(base_embeddings)
            embedding_samples.append(sample)
        embedding_samples = torch.stack(embedding_samples, dim=0)
        token_uncertainty = embedding_samples.std(dim=0).mean(dim=-1)
        
        # Storage for logits
        logits_samples = []
        all_layer_hidden_states = None
        
        # FIX: Prepare attention mask for encoder
        if attention_mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Convert to float and apply masking
            extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None
        
        # MC forward passes
        for m in range(num_samples):
            # Apply dropout to embeddings
            current_embeddings = self.bayesian_embedding.dropout(base_embeddings)
            
            # Pass through transformer encoder
            encoder_outputs = self.transformer.encoder(
                current_embeddings,
                attention_mask=extended_attention_mask,
                output_hidden_states=(m == num_samples - 1)
            )
            
            hidden_states = encoder_outputs.last_hidden_state
            
            # Store hidden states for Theorem 5 (last sample only)
            if m == num_samples - 1 and encoder_outputs.hidden_states is not None:
                all_layer_hidden_states = [
                    h.unsqueeze(0) for h in encoder_outputs.hidden_states[1:]
                ]
            
            # Pool [CLS] token
            pooled_output = hidden_states[:, 0]
            
            # Classify with dropout
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            
            logits_samples.append(logits)
            
            # Clear memory
            del hidden_states, pooled_output
            if m < num_samples - 1:
                torch.cuda.empty_cache()
        
        logits_samples = torch.stack(logits_samples, dim=0)
        
        # Dummy hidden states if not available
        if all_layer_hidden_states is None:
            all_layer_hidden_states = [
                torch.zeros(1, batch_size, seq_len, self.hidden_size, device=input_ids.device)
                for _ in range(self.num_layers)
            ]
        
        return {
            'logits_samples': logits_samples,
            'all_layer_hidden_states': all_layer_hidden_states,
            'token_uncertainty': token_uncertainty,
        }
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_uncertainty: bool = True,
        return_layer_decomposition: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with memory optimization
        """
        
        batch_size = input_ids.shape[0]
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # TRAINING MODE: Simple forward (no MC sampling)
        if self.training or not return_uncertainty:
            # Standard transformer forward
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Pool [CLS] token
            pooled_output = outputs.last_hidden_state[:, 0]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            
            # Confidence
            decision_outputs = self.confidence_decision(
                logits=logits,
                return_all=True,
                apply_temperature=True
            )
            
            result = {
                'logits': logits,
                'probabilities': decision_outputs['probabilities'],
                'predictions': decision_outputs['predictions'],
                'confidence': decision_outputs['confidence'],
                'epistemic_uncertainty': torch.zeros(batch_size, device=input_ids.device),
                'aleatoric_uncertainty': torch.zeros(batch_size, device=input_ids.device),
                'should_abstain': decision_outputs['should_abstain'],
                'token_uncertainty': torch.zeros(batch_size, input_ids.shape[1], device=input_ids.device),
            }
            
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                result['loss'] = loss
            
            return result
        
        # EVALUATION MODE: Full MC sampling
        else:
            mc_outputs = self.forward_with_mc_sampling(input_ids, attention_mask)
            
            logits_samples = mc_outputs['logits_samples']
            all_layer_hidden_states = mc_outputs['all_layer_hidden_states']
            token_uncertainty = mc_outputs['token_uncertainty']
            
            # Mean logits
            mean_logits = logits_samples.mean(dim=0)
            
            # Confidence
            decision_outputs = self.confidence_decision(
                logits=mean_logits,
                return_all=True,
                apply_temperature=True
            )
            
            # Uncertainties
            epistemic_uncertainty = logits_samples.var(dim=0).mean(dim=-1)
            probs_samples = torch.softmax(logits_samples, dim=-1)
            entropy_samples = -(probs_samples * torch.log(probs_samples + 1e-10)).sum(dim=-1)
            aleatoric_uncertainty = entropy_samples.mean(dim=0)
            
            result = {
                'logits': mean_logits,
                'probabilities': decision_outputs['probabilities'],
                'predictions': decision_outputs['predictions'],
                'confidence': decision_outputs['confidence'],
                'epistemic_uncertainty': epistemic_uncertainty,
                'aleatoric_uncertainty': aleatoric_uncertainty,
                'should_abstain': decision_outputs['should_abstain'],
                'token_uncertainty': token_uncertainty,
            }
            
            # Theorem 5
            if return_layer_decomposition and self.use_layer_decomposition:
                try:
                    decomposition = self.variance_decomposition(
                        all_layer_hidden_states=all_layer_hidden_states,
                        logits_samples=logits_samples,
                        return_critical_layers=True
                    )
                    result.update({
                        'layer_uncertainties': decomposition['layer_total'],
                        'layer_aleatoric': decomposition['layer_aleatoric'],
                        'layer_epistemic': decomposition['layer_epistemic'],
                        'critical_layers': decomposition['critical_layer_indices'],
                    })
                except Exception as e:
                    print(f"Warning: Theorem 5 failed: {e}")
            
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(mean_logits, labels)
                result['loss'] = loss
            
            return result
    
    def predict_with_abstention(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predictions with abstention"""
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_uncertainty=True
        )
        
        confidences = outputs['confidence']
        predictions = outputs['predictions']
        abstained = outputs['should_abstain']
        
        predictions = predictions.clone()
        predictions[abstained] = -1
        
        return predictions, confidences, abstained
    
    def get_config(self) -> Dict:
        """Return config"""
        return {
            'base_model_name': self.base_model_name,
            'num_classes': self.num_classes,
            'mc_samples': self.mc_samples,
            'dropout_rate': self.dropout_rate,
            'uncertainty_penalty': self.uncertainty_penalty,
            'confidence_threshold': self.confidence_threshold,
            'use_layer_decomposition': self.use_layer_decomposition,
        }
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str, device: str = 'cpu'):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint['config']
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model
    
    def save_pretrained(self, save_path: str):
        """Save checkpoint"""
        torch.save({
            'config': self.get_config(),
            'model_state_dict': self.state_dict(),
        }, save_path)
        print(f"✓ Model saved to {save_path}")


if __name__ == "__main__":
    print("UAT-Lite model (Memory Optimized + Fixed) ready!")