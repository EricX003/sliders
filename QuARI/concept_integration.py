"""
Integration layer for QuARI concept bottleneck with existing slider search system.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import json

from concept_bottleneck import ConceptBottleneckQuARI
from transformer_hypernetwork import ColumnWiseTransformerHypernetwork
from models import PersonalizedRetrievalModule


class QuARIConceptSystem:
    """
    Complete system integrating QuARI concept bottleneck with slider search.
    """
    
    def __init__(
        self,
        model_checkpoint_path: str,
        vocab_dir: str,
        model_name: str = "openai/clip-vit-large-patch14-336",
        device: str = "cpu",
        guidance_strength: float = 0.1
    ):
        self.device = device
        self.model_name = model_name
        self.guidance_strength = guidance_strength
        
        # Load vocabulary
        self.vocab_embeddings, self.concept_names = self._load_vocabulary(
            vocab_dir, model_name
        )
        
        # Load QuARI model
        self.quari_model = self._load_quari_model(model_checkpoint_path)
        
        # Create concept bottleneck system
        self.concept_bottleneck = ConceptBottleneckQuARI(
            hypernetwork=self.quari_model.hypernetwork,
            vocab_embeddings=self.vocab_embeddings,
            concept_names=self.concept_names,
            guidance_strength=guidance_strength,
            device=device
        )
        
        print(f"✅ QuARI Concept System initialized")
        print(f"   Model: {model_name}")
        print(f"   Vocabulary: {len(self.concept_names)} concepts")
        print(f"   Device: {device}")
    
    def _load_vocabulary(self, vocab_dir: str, model_name: str) -> Tuple[torch.Tensor, List[str]]:
        """Load vocabulary embeddings and concept names."""
        vocab_path = Path(vocab_dir)
        
        # Find vocabulary file for the model
        safe_model_name = model_name.replace('/', '_').replace('-', '_')
        vocab_file = vocab_path / f'vocab_{safe_model_name}.npz'
        
        if not vocab_file.exists():
            # Fallback to generic CLIP large
            vocab_file = vocab_path / 'vocab_openai_clip_vit_large_patch14_336.npz'
        
        if not vocab_file.exists():
            raise FileNotFoundError(f"No vocabulary file found for {model_name}")
        
        # Load vocabulary
        with np.load(vocab_file, allow_pickle=True) as data:
            embeddings = data['embeddings'].astype(np.float32)
            concept_names = data['texts'].tolist()
        
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
        
        print(f"📚 Loaded vocabulary: {len(concept_names)} concepts, dim={embeddings.shape[1]}")
        return embeddings_tensor, concept_names
    
    def _load_quari_model(self, checkpoint_path: str) -> PersonalizedRetrievalModule:
        """Load QuARI model from checkpoint."""
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"QuARI checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint (using our custom loader for dummy models)
        if checkpoint_path.endswith('last.pt'):
            # Load dummy checkpoint
            import pickle
            with open(checkpoint_path, 'rb') as f:
                save_data = pickle.load(f)
            checkpoint = save_data['checkpoint']
        else:
            # Load real PyTorch checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract hyperparameters
        hparams = checkpoint.get('hyperparameters', {})
        
        # Create dummy feature extractor
        class DummyFeatureExtractor:
            def __init__(self):
                self.feature_dim = hparams.get('embedding_dim', 768)
            def parameters(self):
                return []
        
        # Create model with hyperparameters
        model = PersonalizedRetrievalModule(
            feature_extractor=DummyFeatureExtractor(),
            hidden_dim=hparams.get('hidden_dim', 512),
            num_denoising_steps=hparams.get('num_denoising_steps', 4),
            learning_rate=hparams.get('learning_rate', 5e-4),
            low_rank_dim=hparams.get('low_rank_dim', 64),
            using_precomputed_features=hparams.get('using_precomputed_features', True),
            use_separate_decoders=hparams.get('use_separate_decoders', True)
        )
        
        # Load state dict (convert numpy arrays to tensors if needed)
        state_dict = checkpoint['state_dict']
        if isinstance(list(state_dict.values())[0], np.ndarray):
            # Convert numpy arrays to torch tensors
            for key, value in state_dict.items():
                state_dict[key] = torch.from_numpy(value)
        
        # Load only hypernetwork weights
        hypernetwork_state = {k[13:]: v for k, v in state_dict.items() 
                             if k.startswith('hypernetwork.')}
        model.hypernetwork.load_state_dict(hypernetwork_state, strict=False)
        
        model.eval()
        return model
    
    def concept_guided_search(
        self,
        query_text: str,
        concept_weights: Dict[str, float],
        image_embeddings: np.ndarray,
        text_encoder,
        k: int = 10,
        return_analysis: bool = False,
        use_symmetric: bool = True,
        optimization_steps: int = 10,
        lr: float = 0.01
    ) -> Dict[str, Any]:
        """
        Perform concept-guided image search using QuARI.
        
        Args:
            query_text: Input text query
            concept_weights: Dict mapping concept names to weights
            image_embeddings: Gallery image embeddings [N, E]
            text_encoder: Text encoder for query embedding
            k: Number of results to return
            return_analysis: Whether to return detailed concept analysis
            use_symmetric: Whether to apply same transform to both query and gallery
            optimization_steps: Number of gradient steps for QuARI optimization
            lr: Learning rate for QuARI optimization
        """
        with torch.no_grad():
            # Get concept-guided transformations
            inference_result = self.concept_bottleneck.concept_guided_inference(
                query_text=query_text,
                concept_weights=concept_weights,
                text_encoder=text_encoder,
                return_analysis=return_analysis,
                optimization_steps=optimization_steps,
                lr=lr
            )
            
            W_text = inference_result['W_text']
            W_img = inference_result['W_image']
            
            # Apply transformations
            query_emb = text_encoder.extract_text_features([query_text])
            if query_emb.dim() == 3:
                query_emb = query_emb.squeeze(0)
            
            # Transform query
            if use_symmetric:
                # Use same transformation for both query and images
                q_proj = torch.matmul(query_emb.unsqueeze(0), W_text).squeeze(0)
                transform_matrix = W_text  # Use text transform for both
            else:
                # Use separate transformations
                q_proj = torch.matmul(query_emb.unsqueeze(0), W_text).squeeze(0)
                transform_matrix = W_img  # Use image transform for gallery
            
            q_proj = torch.nn.functional.normalize(q_proj, dim=-1)
            
            # Transform image embeddings
            img_emb_tensor = torch.tensor(image_embeddings, dtype=torch.float32, device=self.device)
            img_proj = torch.matmul(img_emb_tensor.unsqueeze(0), transform_matrix).squeeze(0)
            img_proj = torch.nn.functional.normalize(img_proj, dim=-1)
            
            # Compute similarities
            similarities = torch.matmul(q_proj, img_proj.T)
            
            # Get top k results
            top_scores, top_indices = torch.topk(similarities, k)
            
            results = {
                'query': query_text,
                'concept_weights': concept_weights,
                'top_indices': top_indices.cpu().numpy(),
                'top_scores': top_scores.cpu().numpy(),
                'transformation_matrices': {
                    'W_text': W_text.cpu().numpy(),
                    'W_image': W_img.cpu().numpy()
                },
                'use_symmetric': use_symmetric,
                'optimization_steps': optimization_steps,
                'final_loss': inference_result.get('final_loss')
            }
            
            if return_analysis:
                results['concept_analysis'] = inference_result.get('concept_summary')
                results['decomposition_history'] = inference_result.get('decomposition_history')
            
            return results
    
    def analyze_concept_directions(
        self,
        concept_weights: Dict[str, float],
        return_top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze how concepts influence the transformation directions.
        """
        # Create a dummy query for analysis
        dummy_query = torch.randn(1, self.vocab_embeddings.shape[1], device=self.device)
        
        # Convert concept weights
        weight_vector = torch.zeros(len(self.concept_names), device=self.device)
        for concept_name, weight in concept_weights.items():
            if concept_name in self.concept_names:
                idx = self.concept_names.index(concept_name)
                weight_vector[idx] = weight
        
        with torch.no_grad():
            output = self.concept_bottleneck.forward(
                dummy_query,
                concept_weights=weight_vector,
                return_decomposition=True,
                return_all_steps=True
            )
        
        # Analyze final matrices
        W_text = output['W_text']
        W_img = output['W_image']
        
        # SVD analysis
        U_text, S_text, Vt_text = torch.linalg.svd(W_text[0], full_matrices=False)
        U_img, S_img, Vt_img = torch.linalg.svd(W_img[0], full_matrices=False)
        
        # Top directions
        top_dirs_text = Vt_text[:return_top_k]
        top_dirs_img = Vt_img[:return_top_k]
        
        # Compute alignment with vocabulary concepts
        vocab_norm = torch.nn.functional.normalize(self.vocab_embeddings, dim=-1)
        
        direction_analysis = []
        for i in range(return_top_k):
            text_dir = torch.nn.functional.normalize(top_dirs_text[i], dim=-1)
            img_dir = torch.nn.functional.normalize(top_dirs_img[i], dim=-1)
            
            text_sims = torch.matmul(text_dir, vocab_norm.T)
            img_sims = torch.matmul(img_dir, vocab_norm.T)
            combined_sims = (text_sims + img_sims) / 2
            
            top_concept_indices = torch.topk(combined_sims, 3).indices
            
            direction_info = {
                'direction_idx': i,
                'singular_value_text': S_text[i].item(),
                'singular_value_img': S_img[i].item(),
                'top_aligned_concepts': [
                    {
                        'name': self.concept_names[idx.item()],
                        'alignment_score': combined_sims[idx].item(),
                        'input_weight': concept_weights.get(self.concept_names[idx.item()], 0.0)
                    }
                    for idx in top_concept_indices
                ]
            }
            direction_analysis.append(direction_info)
        
        return {
            'active_concepts': {k: v for k, v in concept_weights.items() if abs(v) > 0.01},
            'direction_analysis': direction_analysis,
            'energy_distribution': {
                'text': (S_text / S_text.sum()).cpu().numpy()[:return_top_k],
                'img': (S_img / S_img.sum()).cpu().numpy()[:return_top_k]
            }
        }
    
    def get_concept_suggestions(
        self,
        query_text: str,
        text_encoder,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Suggest relevant concepts for a given query.
        """
        # Encode query
        query_emb = text_encoder.extract_text_features([query_text])
        if query_emb.dim() == 3:
            query_emb = query_emb.squeeze(0)
        
        # Compute similarities with vocabulary
        query_norm = torch.nn.functional.normalize(query_emb, dim=-1)
        vocab_norm = torch.nn.functional.normalize(self.vocab_embeddings, dim=-1)
        
        similarities = torch.matmul(query_norm, vocab_norm.T)
        top_scores, top_indices = torch.topk(similarities, top_k)
        
        suggestions = []
        for score, idx in zip(top_scores, top_indices):
            suggestions.append({
                'concept': self.concept_names[idx.item()],
                'similarity': score.item(),
                'suggested_weight': min(1.0, max(0.1, score.item()))  # Reasonable weight suggestion
            })
        
        return suggestions


def create_quari_concept_system(
    model_checkpoint_path: str,
    vocab_dir: str = "/Users/ericxing/scratch/quari_sliders/vocab",
    model_name: str = "openai/clip-vit-large-patch14-336",
    device: str = "cpu",
    guidance_strength: float = 0.1
) -> QuARIConceptSystem:
    """
    Factory function to create QuARI concept system.
    """
    return QuARIConceptSystem(
        model_checkpoint_path=model_checkpoint_path,
        vocab_dir=vocab_dir,
        model_name=model_name,
        device=device,
        guidance_strength=guidance_strength
    )
