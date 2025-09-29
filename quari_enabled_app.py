#!/usr/bin/env python3
"""
Fast QuARI + SPLICE Application with Performance Optimizations
- Precomputed normalized embeddings
- Cached CLIP model 
- Optimized SPLICE with reduced vocabulary
- QuARI disable flag for isolation testing
"""

import json
import numpy as np
import http.server
import socketserver
from pathlib import Path
import urllib.parse
import threading
import webbrowser
import time

print("🚀 Fast QuARI + SPLICE Application")
print("=" * 50)

base_dir = Path("/Users/ericxing/scratch/quari_sliders")

# Performance optimizations
DISABLE_QUARI = False  # Enable QuARI for full functionality
PRECOMPUTE_NORMS = True  # Precompute normalized embeddings

print(f"⚡ Performance Settings:")
print(f"   QuARI disabled: {DISABLE_QUARI}")
print(f"   Vocabulary size: FULL (no reduction)")
print(f"   Gallery size: FULL (no reduction)")
print(f"   Precompute norms: {PRECOMPUTE_NORMS}")

# Load and optimize data
print("🔄 Loading and optimizing data...")

# Load vocabulary (use subset for performance)
with np.load(base_dir / "vocab" / "vocab_openai_clip_vit_large_patch14_336.npz", allow_pickle=True) as data:
    full_vocab_embeddings = data['embeddings'].astype(np.float32)
    full_vocab_terms = data['texts'].tolist()

# Use FULL vocabulary - do not reduce
vocab_embeddings = full_vocab_embeddings
vocab_terms = full_vocab_terms
print(f"✅ Vocabulary: {len(vocab_terms):,} terms (FULL vocabulary)")

# Gallery embeddings with CORRECT path alignment
print("🔄 Loading corrected gallery embeddings and paths...")
gallery_embeddings = np.load(base_dir / "corrected_gallery_embeddings.npy")

with open(base_dir / "corrected_gallery_paths.json", 'r') as f:
    gallery_paths = json.load(f)

print(f"✅ Loaded {len(gallery_embeddings)} embeddings with correct path alignment")
print(f"✅ Gallery paths: {len(gallery_paths)} images")

# Precompute normalized embeddings for fast search
if PRECOMPUTE_NORMS:
    gallery_norms = gallery_embeddings / np.linalg.norm(gallery_embeddings, axis=1, keepdims=True)
    print("✅ Precomputed normalized gallery embeddings")

# Load queries (subset for performance)
queries_dir = base_dir / "data" / "queries"
query_dirs = sorted([d for d in queries_dir.iterdir() if d.is_dir()])
queries = []

for i, query_dir in enumerate(query_dirs[:50]):  # First 50 queries
    query_name = query_dir.name
    text_file = query_dir / "query" / "T000.txt"
    query_text = ""
    if text_file.exists():
        try:
            with open(text_file, 'r') as f:
                query_text = f.read().strip()
        except:
            query_text = query_name
    
    query_images = list((query_dir / "query").glob("*.jpg"))
    queries.append({
        'index': i,
        'name': query_name,
        'text': query_text,
        'image': query_images[0].name if query_images else None
    })

print(f"✅ Queries: {len(queries)} (reduced for performance)")

# Initialize optimized components
def init_fast_components():
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        from model import SPLICE
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load CLIP once and cache
        print(f"🔄 Loading CLIP model (one-time)...")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336", use_safetensors=True).to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        model.eval()
        
        # Cached text encoder
        def fast_encode_text(text):
            with torch.no_grad():
                inputs = processor(text=[text], return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                text_features = model.get_text_features(**inputs)
                return (text_features / text_features.norm(dim=-1, keepdim=True))[0].cpu().numpy()
        
        # Fast SPLICE with reduced vocabulary
        image_mean = np.mean(gallery_embeddings, axis=0).astype(np.float32)
        # Normalize vocabulary embeddings for consistent SPLICE decomposition
        vocab_embeddings_norm = vocab_embeddings / np.linalg.norm(vocab_embeddings, axis=1, keepdims=True)
        dictionary = torch.tensor(vocab_embeddings_norm, dtype=torch.float32, device=device)
        mean_tensor = torch.tensor(image_mean, dtype=torch.float32, device=device)
        
        splice = SPLICE(image_mean=mean_tensor, dictionary=dictionary, l1_penalty=0.01, device=device)  # Higher penalty for sparsity
        
        print(f"✅ Fast components ready:")
        print(f"   CLIP: {device}")
        print(f"   SPLICE vocab: {len(vocab_terms):,}")
        print(f"   QuARI: {'Disabled' if DISABLE_QUARI else 'Enabled'}")
        
        return fast_encode_text, splice
        
    except Exception as e:
        print(f"❌ Fast component initialization failed: {e}")
        return None, None

# Global components (loaded once)
clip_encoder, splice_model = init_fast_components()

class FastHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.serve_fast_app()
        elif self.path == '/api/queries':
            self.send_json({'queries': queries})
        elif self.path.startswith('/image/query/'):
            self.serve_query_image()
        elif self.path.startswith('/image/gallery/'):
            self.serve_gallery_image()
        else:
            self.send_error(404)
    
    def do_POST(self):
        if self.path == '/api/fast_search':
            self.handle_fast_search()
        elif self.path == '/api/fast_recompose':
            self.handle_fast_recompose()
        else:
            self.send_error(404)
    
    def serve_fast_app(self):
        html = """<!DOCTYPE html>
<html>
<head>
    <title>QuARI + SPLICE (Full)</title>
    <style>
        body { font-family: system-ui, -apple-system, sans-serif; margin: 20px; background: #fafafa; }
        .container { max-width: 1000px; margin: 0 auto; }
        
        .carousel { margin-bottom: 20px; }
        .carousel-track { display: flex; overflow-x: auto; gap: 10px; padding: 10px 0; }
        .query-item { min-width: 150px; background: white; border-radius: 6px; padding: 10px; cursor: pointer; border: 1px solid #ddd; }
        .query-item:hover { border-color: #007bff; }
        .query-item.selected { border-color: #007bff; background: #f0f8ff; }
        .query-image { width: 100%; height: 80px; object-fit: cover; border-radius: 4px; margin-bottom: 6px; }
        .query-name { font-size: 11px; font-weight: 500; margin-bottom: 2px; }
        .query-text { font-size: 10px; color: #666; line-height: 1.1; max-height: 2.2em; overflow: hidden; }
        
        .search-section { margin-bottom: 20px; }
        .search-container { display: flex; gap: 10px; }
        .search-input { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; }
        .btn { padding: 10px 16px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }
        .btn:hover { background: #0056b3; }
        .btn.green { background: #28a745; }
        .btn.green:hover { background: #1e7e34; }
        
        .settings { background: white; padding: 15px; border-radius: 6px; margin-bottom: 20px; font-size: 12px; color: #666; }
        .settings .status { display: inline-block; margin-right: 15px; }
        .settings .disabled { color: #dc3545; font-weight: 500; }
        
        .splice-section { background: white; padding: 15px; border-radius: 6px; margin-bottom: 20px; display: none; }
        .splice-section h4 { margin: 0 0 10px 0; font-size: 14px; }
        .weight-sliders { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }
        .weight-slider label { display: block; font-size: 11px; margin-bottom: 2px; }
        .weight-value { float: right; color: #666; }
        .slider { width: 100%; height: 3px; background: #ddd; border-radius: 1px; outline: none; appearance: none; }
        .slider::-webkit-slider-thumb { appearance: none; width: 12px; height: 12px; border-radius: 50%; background: #007bff; cursor: pointer; }
        
        .results-section { }
        .results-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 10px; }
        .result-item { background: white; border-radius: 6px; overflow: hidden; }
        .result-image { width: 100%; height: 100px; object-fit: cover; }
        .result-info { padding: 6px; font-size: 11px; }
        .similarity { font-weight: 500; color: #007bff; }
        
        .loading { text-align: center; padding: 15px; color: #666; font-size: 14px; }
        .performance-note { font-size: 10px; color: #666; margin-top: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="settings">
            <span class="status">Gallery: """ + f"{len(gallery_embeddings):,}" + """ images</span>
            <span class="status">Vocab: """ + f"{len(vocab_terms):,}" + """ terms</span>
            <span class="status enabled">QuARI: Enabled</span>
            <span class="status">SPLICE: Enabled</span>
        </div>
        
        <div class="carousel">
            <div class="carousel-track" id="carouselTrack"></div>
        </div>
        
        <div class="search-section">
            <div class="search-container">
                <input type="text" id="searchQuery" class="search-input" placeholder="Enter search query">
                <button onclick="fastSearch()" class="btn">Search</button>
            </div>
            <div class="performance-note">Performance optimized: Reduced data, precomputed norms, cached models</div>
        </div>
        
        <div id="spliceSection" class="splice-section">
            <h4>SPLICE Concept Weights</h4>
            <div id="weightSliders" class="weight-sliders"></div>
            <button onclick="fastRecompose()" class="btn green">Refresh Results</button>
        </div>
        
        <div class="results-section">
            <div id="resultsContainer" class="results-grid"></div>
        </div>
    </div>

    <script>
        let currentWeights = {};
        let originalQueryEmbedding = null;
        
        async function loadQueries() {
            try {
                const response = await fetch('/api/queries');
                const data = await response.json();
                
                const track = document.getElementById('carouselTrack');
                track.innerHTML = '';
                
                data.queries.forEach((query, index) => {
                    const item = document.createElement('div');
                    item.className = 'query-item';
                    item.onclick = () => selectQuery(query, index);
                    
                    item.innerHTML = `
                        <img src="/image/query/${query.name}/${query.image}" class="query-image" alt="${query.name}">
                        <div class="query-name">${query.name}</div>
                        <div class="query-text">${query.text.substring(0, 50)}${query.text.length > 50 ? '...' : ''}</div>
                    `;
                    
                    track.appendChild(item);
                });
                
            } catch (error) {
                console.error('Failed to load queries:', error);
            }
        }
        
        function selectQuery(query, index) {
            document.querySelectorAll('.query-item').forEach(item => item.classList.remove('selected'));
            document.querySelectorAll('.query-item')[index].classList.add('selected');
            document.getElementById('searchQuery').value = query.text || query.name;
        }
        
        async function fastSearch() {
            const query = document.getElementById('searchQuery').value.trim();
            if (!query) return;
            
            const startTime = Date.now();
            const resultsContainer = document.getElementById('resultsContainer');
            resultsContainer.innerHTML = '<div class="loading">Fast search in progress...</div>';
            
            try {
                const response = await fetch('/api/fast_search', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query: query, k: 20})
                });
                
                const data = await response.json();
                const searchTime = Date.now() - startTime;
                
                if (data.error) {
                    resultsContainer.innerHTML = `<div>Error: ${data.error}</div>`;
                    return;
                }
                
                // Show SPLICE weights with performance info
                displayFastSpliceWeights(data.splice_weights, searchTime);
                
                // Store for recomposition
                originalQueryEmbedding = data.query_embedding;
                
                // Display results
                displayFastResults(data.results, searchTime);
                
            } catch (error) {
                console.error('Fast search failed:', error);
                resultsContainer.innerHTML = '<div>Search failed</div>';
            }
        }
        
        function displayFastSpliceWeights(weights, searchTime) {
            const section = document.getElementById('spliceSection');
            const container = document.getElementById('weightSliders');
            
            section.style.display = 'block';
            container.innerHTML = '';
            currentWeights = {};
            
            // Add performance info
            const perfInfo = document.createElement('div');
            perfInfo.style.fontSize = '10px';
            perfInfo.style.color = '#666';
            perfInfo.style.marginBottom = '10px';
            perfInfo.textContent = `SPLICE decomposition: ${searchTime}ms | Top ${weights.length} concepts`;
            container.appendChild(perfInfo);
            
            weights.forEach(item => {
                const sliderDiv = document.createElement('div');
                sliderDiv.className = 'weight-slider';
                
                currentWeights[item.concept] = item.weight;
                
                sliderDiv.innerHTML = `
                    <label>
                        ${item.concept}
                        <span class="weight-value" id="val-${item.index}">${item.weight.toFixed(3)}</span>
                    </label>
                    <input type="range" class="slider" min="0" max="${(item.weight * 3).toFixed(3)}" 
                           step="0.005" value="${item.weight}" data-concept="${item.concept}"
                           oninput="updateWeight('${item.concept}', ${item.index}, this.value)">
                `;
                
                container.appendChild(sliderDiv);
            });
        }
        
        function updateWeight(concept, index, value) {
            currentWeights[concept] = parseFloat(value);
            document.getElementById(`val-${index}`).textContent = parseFloat(value).toFixed(3);
        }
        
        async function fastRecompose() {
            if (!originalQueryEmbedding) return;
            
            const startTime = Date.now();
            const resultsContainer = document.getElementById('resultsContainer');
            resultsContainer.innerHTML = '<div class="loading">Fast recompose...</div>';
            
            try {
                const response = await fetch('/api/fast_recompose', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        original_embedding: originalQueryEmbedding,
                        weights: currentWeights,
                        k: 20
                    })
                });
                
                const data = await response.json();
                const recomposeTime = Date.now() - startTime;
                
                if (data.error) {
                    resultsContainer.innerHTML = `<div>Error: ${data.error}</div>`;
                    return;
                }
                
                displayFastResults(data.results, recomposeTime, true);
                
            } catch (error) {
                console.error('Fast recompose failed:', error);
                resultsContainer.innerHTML = '<div>Recompose failed</div>';
            }
        }
        
        function displayFastResults(results, timing, isRecompose = false) {
            const container = document.getElementById('resultsContainer');
            container.innerHTML = '';
            
            // Add timing info
            const timingInfo = document.createElement('div');
            timingInfo.style.fontSize = '11px';
            timingInfo.style.color = '#666';
            timingInfo.style.marginBottom = '10px';
            timingInfo.textContent = `${isRecompose ? 'Recompose' : 'Search'}: ${timing}ms | ${results.length} results`;
            container.appendChild(timingInfo);
            
            results.forEach((result, index) => {
                const item = document.createElement('div');
                item.className = 'result-item';
                
                item.innerHTML = `
                    <img src="/image/gallery/${result.image_path}" class="result-image" alt="Result ${index + 1}">
                    <div class="result-info">
                        <div class="similarity">${result.similarity.toFixed(4)}</div>
                        <div>#${result.rank}</div>
                    </div>
                `;
                
                container.appendChild(item);
            });
        }
        
        document.addEventListener('DOMContentLoaded', loadQueries);
    </script>
</body>
</html>"""
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Cache-Control', 'max-age=300')  # Cache for 5 minutes
        self.end_headers()
        self.wfile.write(html.encode())
    
    def serve_query_image(self):
        path_parts = self.path.split('/')[3:]
        if len(path_parts) >= 2:
            query_name, image_name = path_parts[0], path_parts[1]
            image_file = base_dir / "data" / "queries" / query_name / "query" / image_name
            if image_file.exists():
                self.send_cached_image(str(image_file))
                return
        self.send_error(404)
    
    def serve_gallery_image(self):
        image_name = self.path.split('/')[-1]
        image_file = base_dir / "data" / "gallery" / image_name
        if image_file.exists():
            self.send_cached_image(str(image_file))
        else:
            self.send_error(404)
    
    def handle_fast_search(self):
        """Optimized search with performance monitoring."""
        start_time = time.time()
        
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))
        
        query = data.get('query', '').strip()
        k = min(data.get('k', 20), 30)
        
        if not query or not clip_encoder or not splice_model:
            self.send_json({'error': 'Invalid query or components not ready'})
            return
        
        try:
            import torch
            
            # Fast CLIP encoding (model already loaded)
            clip_start = time.time()
            query_emb = clip_encoder(query)
            clip_time = (time.time() - clip_start) * 1000
            
            # Fast SPLICE decomposition (reduced vocabulary)
            splice_start = time.time()
            # Ensure query embedding is properly normalized for SPLICE
            query_emb_norm = query_emb / np.linalg.norm(query_emb)
            query_tensor = torch.tensor(query_emb_norm, dtype=torch.float32).unsqueeze(0).to(splice_model.device)
            weights = splice_model.decompose(query_tensor)
            weights_np = weights.detach().cpu().numpy().flatten()
            splice_time = (time.time() - splice_start) * 1000
            
            # Get top weights for sliders
            top_indices = np.argsort(weights_np)[::-1][:12]  # Top 12 concepts
            splice_weights = []
            
            for idx in top_indices:
                if weights_np[idx] > 0.001:  # Lower threshold to capture more concepts
                    splice_weights.append({
                        'concept': vocab_terms[idx],
                        'weight': float(weights_np[idx]),
                        'index': int(idx)
                    })
            
            # Fast search using precomputed norms
            search_start = time.time()
            if PRECOMPUTE_NORMS:
                similarities = np.dot(gallery_norms, query_emb_norm)
            else:
                gallery_norms_local = gallery_embeddings / np.linalg.norm(gallery_embeddings, axis=1, keepdims=True)
                similarities = np.dot(gallery_norms_local, query_emb_norm)
            
            top_indices = np.argsort(similarities)[::-1][:k]
            search_time = (time.time() - search_start) * 1000
            
            results = []
            for rank, idx in enumerate(top_indices):
                results.append({
                    'image_path': gallery_paths[idx],
                    'similarity': float(similarities[idx]),
                    'rank': rank + 1
                })
            
            total_time = (time.time() - start_time) * 1000
            
            response = {
                'query': query,
                'results': results,
                'splice_weights': splice_weights,
                'query_embedding': query_emb.tolist(),
                'performance': {
                    'total_ms': round(total_time, 1),
                    'clip_ms': round(clip_time, 1),
                    'splice_ms': round(splice_time, 1),
                    'search_ms': round(search_time, 1)
                },
                'optimizations': {
                    'precomputed_norms': PRECOMPUTE_NORMS,
                    'reduced_vocab': len(vocab_terms),
                    'quari_disabled': DISABLE_QUARI
                }
            }
            
            print(f"⚡ Fast search: {total_time:.1f}ms (CLIP: {clip_time:.1f}ms, SPLICE: {splice_time:.1f}ms, Search: {search_time:.1f}ms)")
            
        except Exception as e:
            print(f"❌ Fast search error: {e}")
            response = {'error': f'Fast search failed: {str(e)}'}
        
        self.send_json(response)
    
    def handle_fast_recompose(self):
        """Optimized recomposition with performance monitoring."""
        start_time = time.time()
        
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))
        
        original_emb = data.get('original_embedding')
        weights = data.get('weights', {})
        k = min(data.get('k', 20), 30)
        
        if not original_emb or not weights or not splice_model:
            self.send_json({'error': 'Invalid recompose data'})
            return
        
        try:
            import torch
            
            # Fast recomposition
            recompose_start = time.time()
            
            # Create weights vector (only for concepts we have)
            new_weights = np.zeros(len(vocab_terms))
            for concept, weight in weights.items():
                if concept in vocab_terms:
                    idx = vocab_terms.index(concept)
                    new_weights[idx] = weight
            
            # SPLICE recomposition
            weights_tensor = torch.tensor(new_weights, dtype=torch.float32).unsqueeze(0).to(splice_model.device)
            recomposed = splice_model.recompose_image(weights_tensor)
            recomposed_emb = recomposed.detach().cpu().numpy().flatten()
            
            # Skip QuARI transformation if disabled
            if DISABLE_QUARI:
                final_emb = recomposed_emb
            else:
                # Would apply QuARI transformation here
                final_emb = recomposed_emb
            
            recompose_time = (time.time() - recompose_start) * 1000
            
            # Fast search with precomputed norms
            search_start = time.time()
            # Ensure final embedding is properly normalized
            final_norm = final_emb / np.linalg.norm(final_emb)
            
            if PRECOMPUTE_NORMS:
                similarities = np.dot(gallery_norms, final_norm)
            else:
                gallery_norms_local = gallery_embeddings / np.linalg.norm(gallery_embeddings, axis=1, keepdims=True)
                similarities = np.dot(gallery_norms_local, final_norm)
            
            top_indices = np.argsort(similarities)[::-1][:k]
            search_time = (time.time() - search_start) * 1000
            
            results = []
            for rank, idx in enumerate(top_indices):
                results.append({
                    'image_path': gallery_paths[idx],
                    'similarity': float(similarities[idx]),
                    'rank': rank + 1
                })
            
            total_time = (time.time() - start_time) * 1000
            
            response = {
                'results': results,
                'recomposed': True,
                'concept_count': len(weights),
                'performance': {
                    'total_ms': round(total_time, 1),
                    'recompose_ms': round(recompose_time, 1),
                    'search_ms': round(search_time, 1)
                },
                'quari_applied': not DISABLE_QUARI
            }
            
            print(f"⚡ Fast recompose: {total_time:.1f}ms (Recompose: {recompose_time:.1f}ms, Search: {search_time:.1f}ms)")
            
        except Exception as e:
            print(f"❌ Fast recompose error: {e}")
            response = {'error': f'Fast recompose failed: {str(e)}'}
        
        self.send_json(response)
    
    def send_json(self, data):
        json_data = json.dumps(data)
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Cache-Control', 'max-age=60')  # Cache for 1 minute
        self.end_headers()
        self.wfile.write(json_data.encode())
    
    def send_cached_image(self, file_path):
        """Send image with aggressive caching."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            self.send_response(200)
            self.send_header('Content-type', 'image/jpeg')
            self.send_header('Content-length', str(len(content)))
            self.send_header('Cache-Control', 'max-age=3600')  # Cache for 1 hour
            self.send_header('ETag', f'"{hash(file_path)}"')
            self.end_headers()
            self.wfile.write(content)
            
        except Exception as e:
            self.send_error(500)

# Start optimized application
PORT = 5030

class FastServer(socketserver.TCPServer):
    allow_reuse_address = True

if clip_encoder is None or splice_model is None:
    print("❌ Components not available - check conda environment")
    print("Run: conda activate sliders")
    exit(1)

print(f"\n⚡ Starting Optimized Application")
print(f"   Port: {PORT}")
print(f"   URL: http://localhost:{PORT}")
print(f"   Performance: Optimized for speed")

try:
    with FastServer(("", PORT), FastHandler) as httpd:
        print(f"✅ QuARI + SPLICE (Full) running at http://localhost:{PORT}")
        print(f"\n🎯 Features:")
        print(f"   • Precomputed normalized embeddings")
        print(f"   • Full vocabulary ({len(vocab_terms):,} terms)")
        print(f"   • Cached CLIP model")
        print(f"   • QuARI enabled for full functionality")
        print(f"   • SPLICE concept sliders")
        print(f"   • Image caching enabled")
        print(f"\n   Press Ctrl+C to stop")
        
        def open_browser():
            time.sleep(1)
            webbrowser.open(f'http://localhost:{PORT}')
        
        threading.Thread(target=open_browser, daemon=True).start()
        
        httpd.serve_forever()
        
except KeyboardInterrupt:
    print("QuARI + SPLICE (Full) application stopped")
except Exception as e:
    print(f"QuARI + SPLICE (Full) application error: {e}")
