#!/usr/bin/env python3
"""
QuARI + SPLICE Integrated Slider Search App

Features:
- Dynamic QuARI inference with concept guidance
- SPLICE concept decomposition and reconstruction  
- Symmetric transforms for query and gallery
- Real-time slider updates with QuARI re-inference
"""

import os
import sys
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, session
from pathlib import Path
from typing import Dict, Any, List, Optional
import uuid
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from concept_integration import QuARIConceptSystem, create_quari_concept_system
from sharded_data_loader import ShardedDataLoader
from FeatureExtractor import FeatureExtractorFactory
from model import SPLICE


class QuARISpliceSliderApp:
    """Integrated QuARI + SPLICE slider search with dynamic inference."""
    
    def __init__(
        self, 
        base_dir: str,
        quari_checkpoint_path: str,
        device: str = "cpu",
        use_dummy_quari: bool = False
    ):
        self.base_dir = Path(base_dir)
        self.device = device
        self.model_key = 'clip_large'
        self.use_dummy_quari = use_dummy_quari
        
        # Initialize data loader
        print("🔄 Loading sharded data...")
        self.data_loader = ShardedDataLoader(base_dir)
        self.data_loader.load_model_data(self.model_key)
        
        # Initialize text encoder
        print("🔄 Loading CLIP Large encoder...")
        self.text_encoder = FeatureExtractorFactory.create_extractor(
            'openai/clip-vit-large-patch14-336', device
        )
        
        # Initialize SPLICE
        print("🔄 Setting up SPLICE...")
        self.splice_model = None
        self.concept_names = []
        self.setup_splice()
        
        # Initialize QuARI system
        print("🔄 Loading QuARI concept system...")
        self.quari_system = None
        self.setup_quari(quari_checkpoint_path)
        
        # Session storage for dynamic updates
        self.session_data = {}
        
        print(f"🎉 QuARI + SPLICE Slider App ready!")
        print(f"   Gallery images: {len(self.data_loader.embeddings[self.model_key]):,}")
        print(f"   SPLICE available: {self.splice_model is not None}")
        print(f"   QuARI available: {self.quari_system is not None}")
    
    def setup_splice(self):
        """Setup SPLICE model for concept decomposition."""
        try:
            vocab_path = self.base_dir / 'vocab' / 'vocab_openai_clip_vit_large_patch14_336.npz'
            
            if not vocab_path.exists():
                print("⚠️ No CLIP Large vocabulary found - SPLICE disabled")
                return
            
            # Load vocabulary
            with np.load(vocab_path, allow_pickle=True) as data:
                vocab_embeddings = data['embeddings'].astype(np.float32)
                self.concept_names = data['texts'].tolist()
            
            # Compute image mean from gallery embeddings
            gallery_embeddings = self.data_loader.embeddings[self.model_key]
            image_mean = np.mean(gallery_embeddings, axis=0).astype(np.float32)
            
            # Create SPLICE model
            dictionary = torch.tensor(vocab_embeddings, dtype=torch.float32, device=self.device)
            mean = torch.tensor(image_mean, dtype=torch.float32, device=self.device)
            
            self.splice_model = SPLICE(
                image_mean=mean,
                dictionary=dictionary,
                l1_penalty=0.005,
                device=self.device
            )
            
            print(f"✅ SPLICE initialized with {len(self.concept_names):,} concepts")
            
        except Exception as e:
            print(f"❌ Failed to setup SPLICE: {e}")
            self.splice_model = None
    
    def setup_quari(self, checkpoint_path: str):
        """Setup QuARI concept system."""
        try:
            if self.use_dummy_quari or not Path(checkpoint_path).exists():
                print("⚠️ Using mock QuARI system (no checkpoint available)")
                self.quari_system = MockQuARISystem(
                    vocab_embeddings=self.splice_model.dictionary if self.splice_model else None,
                    concept_names=self.concept_names,
                    device=self.device
                )
            else:
                self.quari_system = create_quari_concept_system(
                    model_checkpoint_path=checkpoint_path,
                    vocab_dir=str(self.base_dir / 'vocab'),
                    device=self.device
                )
            
            print(f"✅ QuARI system ready")
            
        except Exception as e:
            print(f"❌ Failed to setup QuARI: {e}")
            # Fallback to mock system
            self.quari_system = MockQuARISystem(
                vocab_embeddings=self.splice_model.dictionary if self.splice_model else None,
                concept_names=self.concept_names,
                device=self.device
            )
    
    def search_with_quari_splice(
        self,
        query_text: str,
        concept_weights: Dict[str, float],
        session_id: str,
        k: int = 10,
        use_symmetric: bool = True,
        alpha: float = 0.5,
        optimization_steps: int = 10,
        lr: float = 0.01
    ) -> Dict[str, Any]:
        """
        Integrated search using QuARI + SPLICE.
        
        Process:
        1. SPLICE decomposes query to get concept representation
        2. QuARI generates concept-guided transformations  
        3. Apply symmetric transforms to query and gallery
        4. Blend SPLICE reconstruction with original query
        """
        try:
            # Step 1: SPLICE decomposition of original query
            query_emb = self.text_encoder.extract_text_features([query_text])
            if query_emb.dim() == 3:
                query_emb = query_emb.squeeze(0)
            
            splice_reconstruction = None
            if self.splice_model and len(concept_weights) > 0:
                # Get concept weights as tensor
                weight_vector = torch.zeros(len(self.concept_names), device=self.device)
                for concept_name, weight in concept_weights.items():
                    if concept_name in self.concept_names:
                        idx = self.concept_names.index(concept_name)
                        weight_vector[idx] = weight
                
                # SPLICE reconstruction
                splice_reconstruction = self.splice_model.recompose_image(
                    weight_vector.unsqueeze(0)
                ).squeeze(0)
                
                # Blend original query with SPLICE reconstruction
                query_emb_norm = torch.nn.functional.normalize(query_emb, dim=-1)
                splice_norm = torch.nn.functional.normalize(splice_reconstruction, dim=-1)
                blended_query = alpha * query_emb_norm + (1 - alpha) * splice_norm
                query_emb = torch.nn.functional.normalize(blended_query, dim=-1)
            
            # Step 2: QuARI concept-guided transformation
            gallery_embeddings = self.data_loader.embeddings[self.model_key]
            
            if self.quari_system and len(concept_weights) > 0:
                # Use QuARI for concept-guided search
                result = self.quari_system.concept_guided_search(
                    query_text=query_text,
                    concept_weights=concept_weights,
                    image_embeddings=gallery_embeddings,
                    text_encoder=self.text_encoder,
                    k=k,
                    return_analysis=True,
                    use_symmetric=use_symmetric,
                    optimization_steps=optimization_steps,
                    lr=lr
                )
                
                search_results = []
                for i, (idx, score) in enumerate(zip(result['top_indices'], result['top_scores'])):
                    search_results.append({
                        'image_path': self.data_loader.image_paths[idx],
                        'similarity': float(score),
                        'rank': i + 1
                    })
                
                # Store session data for dynamic updates
                self.session_data[session_id] = {
                    'query_text': query_text,
                    'concept_weights': concept_weights,
                    'original_query_emb': self.text_encoder.extract_text_features([query_text]).detach().cpu().numpy(),
                    'splice_reconstruction': splice_reconstruction.detach().cpu().numpy() if splice_reconstruction is not None else None,
                    'blended_query_emb': query_emb.detach().cpu().numpy(),
                    'transformation_matrices': result['transformation_matrices'],
                    'quari_analysis': result.get('concept_analysis')
                }
                
                return {
                    'query': query_text,
                    'concept_weights': concept_weights,
                    'results': search_results,
                    'total': len(search_results),
                    'mode': 'quari_splice',
                    'use_symmetric': use_symmetric,
                    'alpha': alpha,
                    'optimization_steps': optimization_steps,
                    'final_loss': result.get('final_loss'),
                    'quari_analysis': result.get('concept_analysis'),
                    'splice_used': splice_reconstruction is not None
                }
            else:
                # Fallback to standard search
                return self._standard_search(query_text, k)
                
        except Exception as e:
            print(f"❌ QuARI+SPLICE search error: {e}")
            import traceback
            traceback.print_exc()
            return self._standard_search(query_text, k)
    
    def update_sliders_dynamic(
        self,
        concept_weights: Dict[str, float],
        session_id: str,
        k: int = 10,
        use_symmetric: bool = True,
        alpha: float = 0.5,
        optimization_steps: int = 5,  # Fewer steps for real-time updates
        lr: float = 0.01
    ) -> Dict[str, Any]:
        """
        Dynamic slider update with QuARI re-inference.
        
        As user moves sliders, re-run QuARI optimization with new concept weights.
        """
        try:
            if session_id not in self.session_data:
                return {'error': 'No active session found'}
            
            session_info = self.session_data[session_id]
            query_text = session_info['query_text']
            
            query_emb = torch.tensor(
                session_info['original_query_emb'], 
                dtype=torch.float32, 
                device=self.device
            ).squeeze(0)
            
            splice_reconstruction = None
            if self.splice_model and len(concept_weights) > 0:
                weight_vector = torch.zeros(len(self.concept_names), device=self.device)
                for concept_name, weight in concept_weights.items():
                    if concept_name in self.concept_names:
                        idx = self.concept_names.index(concept_name)
                        weight_vector[idx] = weight
                
                splice_reconstruction = self.splice_model.recompose_image(
                    weight_vector.unsqueeze(0)
                ).squeeze(0)
                
                query_emb_norm = torch.nn.functional.normalize(query_emb, dim=-1)
                splice_norm = torch.nn.functional.normalize(splice_reconstruction, dim=-1)
                blended_query = alpha * query_emb_norm + (1 - alpha) * splice_norm
                query_emb = torch.nn.functional.normalize(blended_query, dim=-1)
            
            # Re-run QuARI inference with updated concept weights
            gallery_embeddings = self.data_loader.embeddings[self.model_key]
            
            result = self.quari_system.concept_guided_search(
                query_text=query_text,
                concept_weights=concept_weights,
                image_embeddings=gallery_embeddings,
                text_encoder=self.text_encoder,
                k=k,
                return_analysis=False,  # Skip analysis for speed
                use_symmetric=use_symmetric,
                optimization_steps=optimization_steps,
                lr=lr
            )
            
            search_results = []
            for i, (idx, score) in enumerate(zip(result['top_indices'], result['top_scores'])):
                search_results.append({
                    'image_path': self.data_loader.image_paths[idx],
                    'similarity': float(score),
                    'rank': i + 1
                })
            
            # Update session data
            self.session_data[session_id].update({
                'concept_weights': concept_weights,
                'splice_reconstruction': splice_reconstruction.detach().cpu().numpy() if splice_reconstruction is not None else None,
                'blended_query_emb': query_emb.detach().cpu().numpy(),
                'transformation_matrices': result['transformation_matrices']
            })
            
            return {
                'results': search_results,
                'total': len(search_results),
                'concept_weights': concept_weights,
                'alpha': alpha,
                'use_symmetric': use_symmetric,
                'final_loss': result.get('final_loss'),
                'optimization_steps': optimization_steps
            }
            
        except Exception as e:
            print(f"❌ Dynamic slider update error: {e}")
            return {'error': str(e)}
    
    def _standard_search(self, query_text: str, k: int) -> Dict[str, Any]:
        """Fallback standard search."""
        text_embedding = self.text_encoder.extract_text_features([query_text])
        text_embedding_np = text_embedding.detach().cpu().numpy().flatten()
        
        results = self.data_loader.search_similar_images(text_embedding_np, self.model_key, k)
        
        return {
            'query': query_text,
            'results': results,
            'total': len(results),
            'mode': 'standard'
        }
    
    def get_concept_suggestions(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get concept suggestions using both SPLICE and QuARI."""
        suggestions = []
        
        if self.quari_system:
            try:
                quari_suggestions = self.quari_system.get_concept_suggestions(
                    query_text, self.text_encoder, top_k
                )
                suggestions.extend(quari_suggestions)
            except Exception as e:
                print(f"❌ QuARI suggestions error: {e}")
        
        # Add SPLICE-based suggestions if available
        if self.splice_model:
            try:
                query_emb = self.text_encoder.extract_text_features([query_text])
                if query_emb.dim() == 3:
                    query_emb = query_emb.squeeze(0)
                
                # SPLICE decomposition to get concept weights
                weights = self.splice_model.decompose(query_emb.unsqueeze(0))
                weights_np = weights.detach().cpu().numpy().flatten()
                
                # Get top concept weights from SPLICE
                top_indices = np.argsort(weights_np)[-top_k:][::-1]
                
                for idx in top_indices:
                    if idx < len(self.concept_names) and weights_np[idx] > 0.01:
                        suggestions.append({
                            'concept': self.concept_names[idx],
                            'similarity': float(weights_np[idx]),
                            'suggested_weight': min(1.0, float(weights_np[idx]) * 2),
                            'source': 'splice'
                        })
                        
            except Exception as e:
                print(f"❌ SPLICE suggestions error: {e}")
        
        # Deduplicate and sort
        seen_concepts = set()
        unique_suggestions = []
        for suggestion in suggestions:
            concept = suggestion['concept']
            if concept not in seen_concepts:
                seen_concepts.add(concept)
                unique_suggestions.append(suggestion)
        
        # Sort by similarity score
        unique_suggestions.sort(key=lambda x: x['similarity'], reverse=True)
        
        return unique_suggestions[:top_k]
    
    def get_status(self) -> Dict[str, Any]:
        """Get app status."""
        model_info = self.data_loader.get_model_info(self.model_key)
        
        return {
            'app': 'quari_splice_slider_search',
            'model': 'CLIP Large + QuARI + SPLICE',
            'device': self.device,
            'gallery_images': model_info['total_embeddings'],
            'embedding_dim': model_info['embedding_dim'],
            'splice_available': self.splice_model is not None,
            'quari_available': self.quari_system is not None,
            'concept_count': len(self.concept_names),
            'ready': model_info['has_index']
        }


class MockQuARISystem:
    """Mock QuARI system for testing when no trained model is available."""
    
    def __init__(self, vocab_embeddings, concept_names, device):
        self.vocab_embeddings = vocab_embeddings
        self.concept_names = concept_names
        self.device = device
    
    def concept_guided_search(self, query_text, concept_weights, image_embeddings, text_encoder, k=10, **kwargs):
        """Mock concept-guided search that returns random results."""
        print(f"🔧 Mock QuARI search: '{query_text}' with {len(concept_weights)} concepts")
        
        # Get random indices
        num_images = len(image_embeddings)
        indices = np.random.choice(num_images, min(k, num_images), replace=False)
        scores = np.random.rand(len(indices))
        
        # Sort by score descending
        sorted_idx = np.argsort(scores)[::-1]
        
        return {
            'top_indices': indices[sorted_idx],
            'top_scores': scores[sorted_idx],
            'transformation_matrices': {
                'W_text': np.random.randn(768, 768),
                'W_image': np.random.randn(768, 768)
            },
            'concept_analysis': {
                'active_concepts': concept_weights,
                'mock_analysis': True
            },
            'final_loss': np.random.rand() * 0.1
        }
    
    def get_concept_suggestions(self, query_text, text_encoder, top_k=10):
        """Mock concept suggestions."""
        if not self.concept_names:
            return []
        
        # Return random concepts
        selected = np.random.choice(
            len(self.concept_names), 
            min(top_k, len(self.concept_names)), 
            replace=False
        )
        
        suggestions = []
        for idx in selected:
            suggestions.append({
                'concept': self.concept_names[idx],
                'similarity': np.random.rand(),
                'suggested_weight': np.random.rand(),
                'source': 'mock_quari'
            })
        
        return suggestions


# Flask app setup
app = Flask(__name__)
app.secret_key = 'quari_splice_slider_secret_key_2024'

# Configuration
BASE_DIR = '/Users/ericxing/scratch/quari_sliders'
QUARI_CHECKPOINT = '/Users/ericxing/scratch/quari_sliders/checkpoints/last.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize integrated app
print("🚀 Initializing QuARI + SPLICE Slider App...")
integrated_app = QuARISpliceSliderApp(
    BASE_DIR, 
    QUARI_CHECKPOINT, 
    DEVICE,
)


@app.route('/')
def index():
    """Main QuARI + SPLICE interface"""
    return render_template('quari_splice_slider.html')


@app.route('/api/status')
def api_status():
    """Get system status"""
    return jsonify(integrated_app.get_status())


@app.route('/api/search_quari_splice', methods=['POST'])
def api_search_quari_splice():
    """Integrated QuARI + SPLICE search"""
    data = request.json
    query = data.get('query', '').strip()
    concept_weights = data.get('concept_weights', {})
    k = min(int(data.get('k', 10)), 20)
    use_symmetric = data.get('use_symmetric', True)
    alpha = float(data.get('alpha', 1))
    optimization_steps = int(data.get('optimization_steps', 10))
    lr = float(data.get('lr', 0.01))
    
    if not query:
        return jsonify({'error': 'Query cannot be empty'}), 400
    
    # Generate session ID
    session_id = session.get('session_id', str(uuid.uuid4()))
    session['session_id'] = session_id
    
    print(f"🔍 QuARI+SPLICE search: '{query}' with {len(concept_weights)} concepts")
    
    result = integrated_app.search_with_quari_splice(
        query, concept_weights, session_id, k, use_symmetric, alpha, optimization_steps, lr
    )
    
    if 'error' in result:
        return jsonify(result), 400
    
    return jsonify(result)


@app.route('/api/update_sliders', methods=['POST'])
def api_update_sliders():
    """Dynamic slider updates with QuARI re-inference"""
    data = request.json
    concept_weights = data.get('concept_weights', {})
    k = min(int(data.get('k', 10)), 20)
    use_symmetric = data.get('use_symmetric', True)
    alpha = float(data.get('alpha', 0.5))
    optimization_steps = int(data.get('optimization_steps', 5))  # Fewer for real-time
    lr = float(data.get('lr', 0.01))
    
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': 'No active session'}), 400
    
    print(f"🎛️ Dynamic slider update: {len(concept_weights)} concepts")
    
    result = integrated_app.update_sliders_dynamic(
        concept_weights, session_id, k, use_symmetric, alpha, optimization_steps, lr
    )
    
    if 'error' in result:
        return jsonify(result), 400
    
    return jsonify(result)


@app.route('/api/concept_suggestions', methods=['POST'])
def api_concept_suggestions():
    """Get concept suggestions from QuARI + SPLICE"""
    data = request.json
    query = data.get('query', '').strip()
    top_k = min(int(data.get('top_k', 10)), 20)
    
    if not query:
        return jsonify({'error': 'Query cannot be empty'}), 400
    
    suggestions = integrated_app.get_concept_suggestions(query, top_k)
    return jsonify({'suggestions': suggestions})


@app.route('/gallery_image/<path:image_path>')
def serve_gallery_image(image_path):
    """Serve gallery images"""
    image_file = integrated_app.base_dir / 'data' / 'gallery' / image_path
    
    if not image_file.exists():
        return jsonify({'error': 'Image not found'}), 404
    
    try:
        return send_file(str(image_file))
    except Exception as e:
        print(f"❌ Error serving image {image_path}: {e}")
        return jsonify({'error': 'Failed to serve image'}), 500


@app.route('/health')
def health():
    """Health check"""
    status = integrated_app.get_status()
    return jsonify({
        'status': 'healthy',
        **status
    })


if __name__ == '__main__':
    print("🚀 Starting QuARI + SPLICE Integrated Slider App")
    print(f"   Model: CLIP Large + QuARI + SPLICE")
    print(f"   Device: {DEVICE}")
    print(f"   Gallery: {integrated_app.get_status()['gallery_images']:,} images")
    print(f"   QuARI: {'✅ Available' if integrated_app.quari_system else '❌ Unavailable'}")
    print(f"   SPLICE: {'✅ Available' if integrated_app.splice_model else '❌ Unavailable'}")
    print("\n🌐 Access at: http://localhost:5004")
    
    app.run(debug=True, host='0.0.0.0', port=5004)
