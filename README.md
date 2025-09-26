# QuARI + SPLICE Concept Slider Search

An advanced image search system combining QuARI (Query-Aligned Retrieval and Inference) with SPLICE for interpretable concept-guided search using dynamic sliders.

## 🎯 Key Features

- **🎛️ Dynamic Concept Sliders**: Real-time adjustment of concept weights with immediate search updates
- **⚡ Gradient-Based Optimization**: True backpropagation through decoder MLPs to optimize latent tokens
- **🔄 QuARI Re-inference**: Dynamic re-optimization as concept sliders are moved
- **🎨 SPLICE Integration**: Concept decomposition and sparse reconstruction 
- **⚖️ Symmetric Transforms**: Option to apply same transformation to both query and gallery
- **📊 Real-time Monitoring**: Live optimization loss tracking and convergence analysis

## 🏗️ Architecture

### Core Components

1. **QuARI Concept Bottleneck** (`QuARI/concept_bottleneck.py`)
   - Gradient-based optimization on intermediate U/V latent tokens
   - Concept-guided inference using vocabulary embeddings
   - Step-by-step decomposition analysis

2. **SPLICE Integration** (`model.py`)
   - Sparse concept decomposition of query embeddings
   - Alpha blending between original and reconstructed queries

3. **Dynamic Slider App** (`QuARI/quari_splice_slider_app.py`)
   - Flask web application with real-time concept controls
   - Session management for optimization state persistence

4. **Web Interface** (`templates/quari_splice_slider.html`)
   - Interactive concept sliders with live weight adjustment
   - Optimization parameter controls and monitoring

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch
- Flask
- NumPy
- scikit-learn

### Installation

```bash
git clone https://github.com/EricX003/sliders.git
cd sliders
pip install torch torchvision flask numpy scikit-learn tqdm
```

### Running the Application

```bash
python run_quari_splice_app.py
```

Access the web interface at: http://localhost:5004

## 💡 How It Works

### 1. Initial Query Processing
- User enters a text query (e.g., "beautiful sunset over mountains")
- System suggests relevant concepts using SPLICE decomposition
- User adjusts concept weights via interactive sliders

### 2. QuARI Concept Guidance
- Target concept representation computed from weighted vocabulary
- Gradient optimization refines U/V tokens to align with concepts:
  ```python
  concept_loss = MSE(transformed_concept, target_concept)
  direction_loss = -cosine_similarity(top_direction, target_concept)
  total_loss = concept_loss + 0.1 * direction_loss + regularization
  ```

### 3. Dynamic Updates
- As sliders move, QuARI re-runs optimization with new concept weights
- Results update in real-time showing concept influence
- SPLICE reconstruction blends with original query

## 🎛️ Configuration Options

### QuARI Optimization
- **optimization_steps**: Number of gradient steps (5-20)
- **lr**: Learning rate for token optimization (0.005-0.05)
- **use_symmetric**: Apply same transform to query and gallery

### SPLICE Blending
- **alpha**: Balance between original and SPLICE reconstruction (0.0-1.0)
- **concept_weights**: Dict mapping concept names to weights (0.0-1.0)

## 📁 Project Structure

```
sliders/
├── QuARI/                           # Core QuARI implementation
│   ├── concept_bottleneck.py        # Gradient-based concept guidance
│   ├── concept_integration.py       # High-level QuARI system interface
│   ├── quari_splice_slider_app.py   # Main Flask application
│   ├── transformer_hypernetwork.py  # Base hypernetwork architecture
│   └── models.py                    # PersonalizedRetrievalModule
├── checkpoints/
│   └── last.pt                      # Dummy QuARI model weights (60.9 MB)
├── templates/
│   └── quari_splice_slider.html     # Web interface
├── vocab/                           # Concept vocabulary embeddings
├── model.py                         # SPLICE implementation
└── run_quari_splice_app.py          # Application runner
```

## 🔬 Technical Details

### Gradient-Based Concept Guidance

Unlike traditional feedforward approaches, this system directly optimizes the intermediate latent representations:

```python
# Make tokens require gradients
u_tok = u_tok.clone().detach().requires_grad_(True)
v_tok = v_tok.clone().detach().requires_grad_(True)

# Setup optimizer on tokens themselves
optimizer = torch.optim.Adam([u_tok, v_tok], lr=lr)

for step in range(optimization_steps):
    # Decode tokens → transformation matrices
    W_text, W_img = hypernetwork._decode_and_proj(u_tok, v_tok)
    
    # Compute concept alignment loss
    loss = concept_alignment_loss(W_text, W_img, target_concepts)
    
    # Backpropagate and update tokens
    loss.backward()
    optimizer.step()
```

### SPLICE + QuARI Integration

The system combines SPLICE's sparse concept decomposition with QuARI's learned transformations:

1. SPLICE decomposes query into interpretable concepts
2. User adjusts concept weights via sliders
3. QuARI generates concept-guided transformations via gradient optimization
4. Search applies transformations to both query and gallery embeddings

## 📊 Performance

- **Latency**: ~50-200ms per optimization cycle
- **Memory**: Moderate additional memory for gradients
- **Quality**: Better concept alignment with more optimization steps
- **Scalability**: Supports vocabularies of 37k+ concepts

## 🔄 API Endpoints

- `POST /api/search_quari_splice` - Initial search with concept weights
- `POST /api/update_sliders` - Dynamic slider updates with re-inference
- `POST /api/concept_suggestions` - Get concept suggestions for queries
- `GET /api/status` - System status and capabilities

## 🎯 Use Cases

- **Research**: Interpretable image retrieval and concept analysis
- **Education**: Understanding concept influence on search results
- **Development**: Prototyping concept-guided search systems
- **Analysis**: Investigating learned representations and transformations

## 📝 Citation

If you use this system in your research, please cite:

```bibtex
@software{quari_splice_sliders,
  title={QuARI + SPLICE Concept Slider Search},
  author={EricX003},
  url={https://github.com/EricX003/sliders},
  year={2024}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- QuARI: Query-Aligned Retrieval and Inference
- SPLICE: Sparse Linear Concept Extraction
- PyTorch Lightning for model training framework
- Flask for web application framework