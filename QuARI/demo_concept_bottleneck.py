"""
Demo script for ConceptBottleneckQuARI.

Loads the transformer_hypernetwork from the checkpoint file and demonstrates
the basic functionality of the ConceptBottleneckQuARI class.
"""

import torch
import torch.nn.functional as F
from transformer_hypernetwork import TransformerHypernetwork
from concept_bottleneck import ConceptBottleneckQuARI


def load_hypernetwork_from_checkpoint(ckpt_path: str, device: str = "cpu") -> TransformerHypernetwork:
    """
    Load only the transformer_hypernetwork weights from a checkpoint file.
    
    Args:
        ckpt_path: Path to the .ckpt file
        device: Device to load the model onto
        
    Returns:
        Loaded TransformerHypernetwork model
    """
    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Get hyperparameters from checkpoint
    hparams = ckpt.get('hyper_parameters', {})
    
    # Create the hypernetwork with the same architecture
    hypernetwork = TransformerHypernetwork(
        embedding_dim=768,  # CLIP large embedding dimension
        low_rank_dim=hparams.get('low_rank_dim', 64),
        hidden_dim=hparams.get('hidden_dim', 512),
        num_steps=hparams.get('num_denoising_steps', 4),
        nhead=hparams.get('nhead', 8),
        num_layers=hparams.get('num_encoder_layers', 6),
        dropout=hparams.get('dropout', 0.1),
        query_residual_weight=hparams.get('query_residual_weight', 0.5),
        anchor_scale=hparams.get('anchor_scale', 0.8),
    )
    
    # Extract only hypernetwork weights from state_dict
    state_dict = ckpt['state_dict']
    hypernetwork_state_dict = {}
    
    for key, value in state_dict.items():
        if key.startswith('hypernetwork.'):
            # Remove the 'hypernetwork.' prefix
            new_key = key[len('hypernetwork.'):]
            hypernetwork_state_dict[new_key] = value
    
    # Load the weights
    hypernetwork.load_state_dict(hypernetwork_state_dict)
    hypernetwork.to(device)
    hypernetwork.eval()
    
    print(f"Successfully loaded hypernetwork with {sum(p.numel() for p in hypernetwork.parameters())} parameters")
    
    return hypernetwork


def create_mock_vocab_embeddings(vocab_size: int, embedding_dim: int, device: str = "cpu") -> tuple:
    """
    Create mock vocabulary embeddings for demonstration.
    In practice, these would come from a real text encoder.
    """
    # Create random normalized embeddings
    vocab_embeddings = torch.randn(vocab_size, embedding_dim, device=device)
    vocab_embeddings = F.normalize(vocab_embeddings, dim=-1)
    
    # Create concept names
    concept_names = [f"concept_{i}" for i in range(vocab_size)]
    
    return vocab_embeddings, concept_names


def demo_basic_forward():
    """Demonstrate basic forward pass through ConceptBottleneckQuARI."""
    print("\n" + "="*60)
    print("Demo 1: Basic Forward Pass")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the hypernetwork
    ckpt_path = "/u/ericx003/sliders/QuARI/weights/weights.ckpt"
    hypernetwork = load_hypernetwork_from_checkpoint(ckpt_path, device)
    
    # Create mock vocabulary
    vocab_size = 100
    embedding_dim = 768
    vocab_embeddings, concept_names = create_mock_vocab_embeddings(
        vocab_size, embedding_dim, device
    )
    
    # Create ConceptBottleneckQuARI
    cbq = ConceptBottleneckQuARI(
        hypernetwork=hypernetwork,
        vocab_embeddings=vocab_embeddings,
        concept_names=concept_names,
        guidance_strength=0.1,
        device=device
    )
    cbq.to(device)
    cbq.eval()
    
    # Create a mock query embedding
    batch_size = 2
    query_emb = torch.randn(batch_size, embedding_dim, device=device)
    query_emb = F.normalize(query_emb, dim=-1)
    
    # Forward pass without concept weights (uses standard hypernetwork)
    print("\n1. Forward pass without concept weights:")
    with torch.no_grad():
        output = cbq(query_emb, concept_weights=None, return_all_steps=True)
    
    print(f"   Output keys: {list(output.keys())}")
    print(f"   W_image shape: {output['W_image'].shape}")
    print(f"   Number of steps returned: {len(output.get('all', []))}")
    
    return cbq, query_emb


def demo_concept_guided_forward(cbq, query_emb):
    """Demonstrate concept-guided forward pass."""
    print("\n" + "="*60)
    print("Demo 2: Concept-Guided Forward Pass")
    print("="*60)
    
    device = query_emb.device
    vocab_size = cbq.vocab_size
    batch_size = query_emb.shape[0]
    
    # Create concept weights (emphasize first few concepts)
    concept_weights = torch.zeros(batch_size, vocab_size, device=device)
    concept_weights[:, 0] = 1.0   # Strongly weight first concept
    concept_weights[:, 1] = 0.5   # Moderately weight second concept
    concept_weights[:, 2] = 0.3   # Slightly weight third concept
    
    print(f"   Applying concept weights to {(concept_weights > 0).sum().item()} concepts")
    
    # Forward pass with concept guidance
    with torch.no_grad():
        output = cbq(
            query_emb,
            concept_weights=concept_weights,
            return_decomposition=True,
            return_all_steps=True,
            optimization_steps=5,
            lr=0.01
        )
    
    print(f"   Output keys: {list(output.keys())}")
    print(f"   W_text shape: {output['W_text'].shape}")
    print(f"   W_image shape: {output['W_image'].shape}")
    print(f"   Optimization steps: {output['optimization_steps']}")
    print(f"   Final loss: {output['final_loss']:.6f}" if output['final_loss'] else "   Final loss: N/A")
    
    return output


def demo_decomposition_analysis(cbq, query_emb):
    """Demonstrate concept decomposition analysis."""
    print("\n" + "="*60)
    print("Demo 3: Decomposition Analysis")
    print("="*60)
    
    device = query_emb.device
    vocab_size = cbq.vocab_size
    batch_size = query_emb.shape[0]
    
    # Create diverse concept weights
    concept_weights = torch.randn(batch_size, vocab_size, device=device)
    concept_weights = F.softmax(concept_weights, dim=-1)
    
    # Run with decomposition analysis
    with torch.no_grad():
        output = cbq(
            query_emb,
            concept_weights=concept_weights,
            return_decomposition=True,
            return_all_steps=True,
            optimization_steps=6,
            lr=0.01
        )
    
    # Analyze the decomposition history
    decomp_history = output.get('decomposition_history', [])
    print(f"   Number of decomposition snapshots: {len(decomp_history)}")
    
    if decomp_history:
        latest_decomp = decomp_history[-1]
        print(f"   Latest decomposition step: {latest_decomp.get('step', 'N/A')}")
        
        if 'singular_values_text' in latest_decomp:
            sv_text = latest_decomp['singular_values_text']
            print(f"   Top singular values (text): {sv_text[0, :3].tolist()}")
        
        if 'energy_concentration' in latest_decomp:
            energy = latest_decomp['energy_concentration']
            print(f"   Energy concentration (text): {energy['text'][0].item():.3f}")
            print(f"   Energy concentration (img): {energy['img'][0].item():.3f}")


def demo_broadcasting():
    """Demonstrate concept weight broadcasting."""
    print("\n" + "="*60)
    print("Demo 4: Concept Weight Broadcasting")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load fresh model
    ckpt_path = "/u/ericx003/sliders/QuARI/weights/weights.ckpt"
    hypernetwork = load_hypernetwork_from_checkpoint(ckpt_path, device)
    
    vocab_size = 50
    embedding_dim = 768
    vocab_embeddings, concept_names = create_mock_vocab_embeddings(
        vocab_size, embedding_dim, device
    )
    
    cbq = ConceptBottleneckQuARI(
        hypernetwork=hypernetwork,
        vocab_embeddings=vocab_embeddings,
        concept_names=concept_names,
        device=device
    )
    cbq.to(device)
    cbq.eval()
    
    # Create batch of queries
    batch_size = 4
    query_emb = F.normalize(torch.randn(batch_size, embedding_dim, device=device), dim=-1)
    
    # Use 1D concept weights (will be broadcast to all batch items)
    concept_weights_1d = torch.zeros(vocab_size, device=device)
    concept_weights_1d[0] = 1.0
    concept_weights_1d[5] = 0.5
    
    print(f"   Batch size: {batch_size}")
    print(f"   Concept weights shape: {concept_weights_1d.shape} (1D - will be broadcast)")
    
    with torch.no_grad():
        output = cbq(
            query_emb,
            concept_weights=concept_weights_1d,
            optimization_steps=3
        )
    
    print(f"   Output W_text shape: {output['W_text'].shape}")
    print(f"   Output W_image shape: {output['W_image'].shape}")
    print("   Broadcasting successful!")


def main():
    """Run all demos."""
    print("="*60)
    print("ConceptBottleneckQuARI Demo Script")
    print("="*60)
    
    # Run demos
    cbq, query_emb = demo_basic_forward()
    demo_concept_guided_forward(cbq, query_emb)
    demo_decomposition_analysis(cbq, query_emb)
    demo_broadcasting()
    
    print("\n" + "="*60)
    print("All demos completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
