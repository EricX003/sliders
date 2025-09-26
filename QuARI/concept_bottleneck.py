import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from transformer_hypernetwork import ColumnWiseTransformerHypernetwork


class ConceptBottleneckQuARI(nn.Module):
    """
    QuARI with concept bottleneck for interpretable inference.
    
    Guides the iterative denoising process using concept weights from vocabulary,
    updating intermediate latent vectors to align with interpretable concepts.
    """
    
    def __init__(
        self,
        hypernetwork: ColumnWiseTransformerHypernetwork,
        vocab_embeddings: torch.Tensor,
        concept_names: List[str],
        concept_projection_dim: int = 256,
        guidance_strength: float = 0.1,
        device: str = "cpu"
    ):
        super().__init__()
        self.hypernetwork = hypernetwork
        self.vocab_embeddings = vocab_embeddings.to(device)  # [vocab_size, embedding_dim]
        self.concept_names = concept_names
        self.guidance_strength = guidance_strength
        self.device = device
        
        self.vocab_size, self.embedding_dim = vocab_embeddings.shape
        
        # No additional learnable parameters needed for gradient-based optimization
        
        # Concept decomposition analyzer
        self.decomp_analyzer = ConceptDecompositionAnalyzer(
            vocab_embeddings, concept_names, device
        )
        
    def forward(
        self,
        query_emb: torch.Tensor,
        concept_weights: Optional[torch.Tensor] = None,
        return_decomposition: bool = False,
        return_all_steps: bool = True,
        optimization_steps: int = 10,
        lr: float = 0.01
    ) -> Dict[str, Any]:
        """
        Concept-guided forward pass with gradient-based optimization.
        
        Args:
            query_emb: [B, E] query embeddings
            concept_weights: [B, vocab_size] or [vocab_size] concept weights
            return_decomposition: Whether to return concept decomposition analysis
            return_all_steps: Whether to return intermediate steps
            optimization_steps: Number of gradient steps to take on latents
            lr: Learning rate for gradient optimization
        """
        B, E = query_emb.shape
        device = query_emb.device
        
        if concept_weights is None:
            # Fallback to standard hypernetwork
            return self.hypernetwork(query_emb, return_all=return_all_steps)
            
        # Handle concept weights broadcasting
        if concept_weights.dim() == 1:
            concept_weights = concept_weights.unsqueeze(0).expand(B, -1)
        
        # Compute target concept representation
        target_concept_repr = self._compute_target_concept_representation(concept_weights)
        
        # Run standard hypernetwork to get initial tokens
        u_tok, v_tok = self._run_standard_denoising(query_emb)
        
        # Optimize tokens via gradient descent
        u_tok, v_tok, optimization_history = self._optimize_tokens_with_concepts(
            u_tok, v_tok, target_concept_repr, optimization_steps, lr,
            return_history=return_all_steps or return_decomposition
        )
        
        # Final decode
        W_text, W_img = self.hypernetwork._decode_and_proj(u_tok, v_tok)
        
        output = {
            'W_text': W_text,
            'W_image': W_img,
            'optimization_steps': optimization_steps,
            'final_loss': optimization_history[-1]['loss'] if optimization_history else None
        }
        
        if return_all_steps:
            output['all'] = optimization_history
            
        if return_decomposition:
            output['decomposition_history'] = [step['decomposition'] for step in optimization_history if 'decomposition' in step]
            
        return output
    
    def _compute_target_concept_representation(self, concept_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute target concept representation from weighted vocabulary.
        
        Args:
            concept_weights: [B, vocab_size] weights for each concept
            
        Returns:
            target_repr: [B, E] target representation in embedding space
        """
        B = concept_weights.shape[0]
        
        # Normalize concept weights
        concept_weights_norm = F.softmax(concept_weights, dim=-1)
        
        # Weighted combination of vocabulary embeddings - this is our target
        target_repr = torch.matmul(concept_weights_norm, self.vocab_embeddings)  # [B, E]
        
        return target_repr
    
    def _run_standard_denoising(self, query_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run standard hypernetwork denoising to get initial token states.
        
        Args:
            query_emb: [B, E] query embeddings
            
        Returns:
            u_tok, v_tok: [B, r, H] initial token states
        """
        B, E = query_emb.shape
        device = query_emb.device
        
        # Encode query
        ctx = self.hypernetwork.query_encoder(query_emb)  # [B, H]
        
        # Initialize tokens
        u_tok = torch.zeros(B, self.hypernetwork.r, self.hypernetwork.H, device=device)
        v_tok = torch.zeros(B, self.hypernetwork.r, self.hypernetwork.H, device=device)
        
        # Standard denoising steps
        for t in range(self.hypernetwork.num_steps):
            step_idx = torch.full((B,), t, device=device, dtype=torch.long)
            cond = (ctx + self.hypernetwork.step_embedding(step_idx)).unsqueeze(1)
            
            # Add positional encoding
            from transformer_hypernetwork import scaled_positional_encoding
            u_seq = scaled_positional_encoding(u_tok, self.hypernetwork.pos_emb)
            v_seq = scaled_positional_encoding(v_tok, self.hypernetwork.pos_emb)
            
            # Build sequence and run transformer
            seq = torch.cat([cond, u_seq, v_seq], dim=1)
            out = self.hypernetwork.transformer(seq)
            delta = out[:, 1:, :]
            
            # Update tokens
            d_u = delta[:, :self.hypernetwork.r, :]
            d_v = delta[:, self.hypernetwork.r:, :]
            u_tok = u_tok + d_u
            v_tok = v_tok + d_v
        
        return u_tok, v_tok
    
    def _optimize_tokens_with_concepts(
        self,
        u_tok: torch.Tensor,
        v_tok: torch.Tensor,
        target_concept_repr: torch.Tensor,
        optimization_steps: int,
        lr: float,
        return_history: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Optimize token representations via gradient descent to align with concepts.
        
        Args:
            u_tok, v_tok: [B, r, H] initial token states
            target_concept_repr: [B, E] target concept representation
            optimization_steps: Number of gradient steps
            lr: Learning rate
            return_history: Whether to return optimization history
            
        Returns:
            optimized u_tok, v_tok and optimization history
        """
        # Make tokens require gradients
        u_tok = u_tok.clone().detach().requires_grad_(True)
        v_tok = v_tok.clone().detach().requires_grad_(True)
        
        # Setup optimizer
        optimizer = torch.optim.Adam([u_tok, v_tok], lr=lr)
        
        history = []
        
        for step in range(optimization_steps):
            optimizer.zero_grad()
            
            # Decode current tokens to get transformation matrices
            W_text, W_img = self.hypernetwork._decode_and_proj(u_tok, v_tok)
            
            # Apply transformations to target concept representation
            # We want the transformed concept to align with the original concept
            transformed_concept = torch.bmm(
                target_concept_repr.unsqueeze(1), W_text
            ).squeeze(1)  # [B, E]
            
            # Concept alignment loss - transformed concept should be similar to target
            concept_loss = F.mse_loss(
                F.normalize(transformed_concept, dim=-1),
                F.normalize(target_concept_repr, dim=-1)
            )
            
            # Direction alignment loss - encourage interpretable directions
            direction_loss = self._compute_direction_alignment_loss(W_text, W_img, target_concept_repr)
            
            # Regularization - prevent tokens from growing too large
            reg_loss = 0.001 * (torch.norm(u_tok) + torch.norm(v_tok))
            
            total_loss = concept_loss + 0.1 * direction_loss + reg_loss
            
            # Backpropagate and update tokens
            total_loss.backward()
            optimizer.step()
            
            if return_history:
                step_info = {
                    'step': step,
                    'loss': total_loss.item(),
                    'concept_loss': concept_loss.item(),
                    'direction_loss': direction_loss.item(),
                    'W_text': W_text.detach().clone(),
                    'W_image': W_img.detach().clone()
                }
                
                # Add decomposition analysis if needed
                if step % 2 == 0:  # Every other step to save compute
                    step_info['decomposition'] = self._analyze_step_decomposition(
                        W_text.detach(), W_img.detach(), 
                        torch.ones(target_concept_repr.shape[0], self.vocab_size, device=target_concept_repr.device) / self.vocab_size,
                        step
                    )
                
                history.append(step_info)
        
        return u_tok.detach(), v_tok.detach(), history
    
    def _compute_direction_alignment_loss(
        self, 
        W_text: torch.Tensor, 
        W_img: torch.Tensor, 
        target_concept_repr: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss encouraging transformation directions to align with concepts.
        
        This encourages the top singular vectors of the transformation matrices
        to align with the target concept representation.
        """
        B = W_text.shape[0]
        
        # Get top singular vectors
        try:
            U_text, S_text, Vt_text = torch.linalg.svd(W_text, full_matrices=False)
            U_img, S_img, Vt_img = torch.linalg.svd(W_img, full_matrices=False)
            
            # Top direction should align with target concept
            top_dir_text = Vt_text[:, 0, :]  # [B, E] - top singular vector
            top_dir_img = Vt_img[:, 0, :]
            
            # Compute alignment with target concept
            target_norm = F.normalize(target_concept_repr, dim=-1)
            dir_text_norm = F.normalize(top_dir_text, dim=-1)
            dir_img_norm = F.normalize(top_dir_img, dim=-1)
            
            # We want high cosine similarity (low negative cosine similarity)
            alignment_loss = -F.cosine_similarity(dir_text_norm, target_norm, dim=-1).mean() \
                           - F.cosine_similarity(dir_img_norm, target_norm, dim=-1).mean()
            
            return alignment_loss
            
        except torch.linalg.LinAlgError:
            # Fallback if SVD fails
            return torch.tensor(0.0, device=W_text.device)
    
    def _analyze_step_decomposition(
        self, 
        W_text: torch.Tensor, 
        W_img: torch.Tensor,
        concept_weights: torch.Tensor,
        step: int
    ) -> Dict[str, Any]:
        """
        Analyze the decomposition at each step to understand concept alignment.
        """
        B = W_text.shape[0]
        
        # Compute dominant directions in W_text and W_img
        U_text, S_text, Vt_text = torch.linalg.svd(W_text, full_matrices=False)
        U_img, S_img, Vt_img = torch.linalg.svd(W_img, full_matrices=False)
        
        # Take top directions
        top_k = min(5, S_text.shape[1])
        top_dirs_text = Vt_text[:, :top_k, :]  # [B, top_k, E]
        top_dirs_img = Vt_img[:, :top_k, :]
        
        # Analyze alignment with vocabulary concepts
        concept_alignment = self.decomp_analyzer.analyze_direction_alignment(
            top_dirs_text, top_dirs_img, concept_weights
        )
        
        return {
            'step': step,
            'singular_values_text': S_text[:, :top_k].detach(),
            'singular_values_img': S_img[:, :top_k].detach(),
            'concept_alignment': concept_alignment,
            'energy_concentration': {
                'text': (S_text[:, :top_k].sum(dim=1) / S_text.sum(dim=1)).detach(),
                'img': (S_img[:, :top_k].sum(dim=1) / S_img.sum(dim=1)).detach()
            }
        }
    
    def concept_guided_inference(
        self,
        query_text: str,
        concept_weights: Dict[str, float],
        text_encoder,
        return_analysis: bool = True,
        optimization_steps: int = 10,
        lr: float = 0.01
    ) -> Dict[str, Any]:
        """
        High-level interface for concept-guided inference with gradient optimization.
        
        Args:
            query_text: Input text query
            concept_weights: Dict mapping concept names to weights
            text_encoder: Text encoder to embed query
            return_analysis: Whether to return detailed analysis
            optimization_steps: Number of gradient steps for token optimization
            lr: Learning rate for optimization
        """
        # Encode query
        query_emb = text_encoder.extract_text_features([query_text])
        if query_emb.dim() == 3:
            query_emb = query_emb.squeeze(0)
        
        # Convert concept weights to tensor
        weight_vector = torch.zeros(len(self.concept_names), device=self.device)
        for concept_name, weight in concept_weights.items():
            if concept_name in self.concept_names:
                idx = self.concept_names.index(concept_name)
                weight_vector[idx] = weight
        
        # Run gradient-based inference
        output = self.forward(
            query_emb,
            concept_weights=weight_vector,
            return_decomposition=return_analysis,
            return_all_steps=return_analysis,
            optimization_steps=optimization_steps,
            lr=lr
        )
        
        if return_analysis:
            output['concept_summary'] = self._summarize_optimization_process(
                output.get('all', []), concept_weights
            )
        
        return output
    
    def _summarize_optimization_process(
        self, 
        optimization_history: List[Dict], 
        concept_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Summarize how gradient optimization aligned tokens with concepts."""
        active_concepts = {k: v for k, v in concept_weights.items() if abs(v) > 0.01}
        
        if not optimization_history:
            return {'active_concepts': active_concepts, 'optimization_steps': 0}
        
        # Track loss evolution
        loss_evolution = [step['loss'] for step in optimization_history]
        concept_loss_evolution = [step['concept_loss'] for step in optimization_history]
        
        # Get final decomposition if available
        final_decomposition = None
        for step in reversed(optimization_history):
            if 'decomposition' in step:
                final_decomposition = step['decomposition']
                break
        
        return {
            'active_concepts': active_concepts,
            'optimization_steps': len(optimization_history),
            'loss_evolution': loss_evolution,
            'concept_loss_evolution': concept_loss_evolution,
            'initial_loss': loss_evolution[0] if loss_evolution else None,
            'final_loss': loss_evolution[-1] if loss_evolution else None,
            'loss_reduction': (loss_evolution[0] - loss_evolution[-1]) if len(loss_evolution) > 1 else 0,
            'final_decomposition': final_decomposition
        }


class ConceptDecompositionAnalyzer:
    """Helper class to analyze concept alignment in matrix decompositions."""
    
    def __init__(self, vocab_embeddings: torch.Tensor, concept_names: List[str], device: str):
        self.vocab_embeddings = vocab_embeddings.to(device)
        self.concept_names = concept_names
        self.device = device
        
    def analyze_direction_alignment(
        self, 
        directions_text: torch.Tensor, 
        directions_img: torch.Tensor,
        concept_weights: torch.Tensor,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Analyze how matrix directions align with vocabulary concepts.
        
        Args:
            directions_text: [B, num_dirs, E] top singular vectors from text matrix
            directions_img: [B, num_dirs, E] top singular vectors from image matrix  
            concept_weights: [B, vocab_size] input concept weights
            top_k: Number of top alignments to return
        """
        B, num_dirs, E = directions_text.shape
        
        # Normalize directions and vocabulary
        dirs_text_norm = F.normalize(directions_text, dim=-1)
        dirs_img_norm = F.normalize(directions_img, dim=-1)
        vocab_norm = F.normalize(self.vocab_embeddings, dim=-1)
        
        alignments = []
        
        for b in range(B):
            batch_alignments = []
            
            for d in range(num_dirs):
                # Compute alignment with vocabulary
                text_sim = torch.matmul(dirs_text_norm[b, d], vocab_norm.T)  # [vocab_size]
                img_sim = torch.matmul(dirs_img_norm[b, d], vocab_norm.T)
                
                # Combined alignment score
                combined_sim = (text_sim + img_sim) / 2
                
                # Weight by input concept weights
                weighted_sim = combined_sim * concept_weights[b]
                
                # Get top alignments
                top_indices = torch.topk(weighted_sim, top_k).indices
                top_scores = weighted_sim[top_indices]
                
                direction_info = {
                    'direction_idx': d,
                    'top_concepts': [
                        {
                            'name': self.concept_names[idx.item()],
                            'score': score.item(),
                            'text_sim': text_sim[idx].item(),
                            'img_sim': img_sim[idx].item()
                        }
                        for idx, score in zip(top_indices, top_scores)
                    ]
                }
                batch_alignments.append(direction_info)
            
            alignments.append(batch_alignments)
        
        return alignments[0] if B == 1 else alignments
