#!/usr/bin/env python3
"""
Main runner for QuARI + SPLICE Slider App

This script starts the integrated QuARI + SPLICE concept slider search application.
"""

import sys
from pathlib import Path

# Add QuARI directory to path
sys.path.append(str(Path(__file__).parent / "QuARI"))

try:
    from QuARI.quari_splice_slider_app import app, integrated_app
    
    if __name__ == '__main__':
        print("🚀 Starting QuARI + SPLICE Integrated Slider App")
        print("=" * 60)
        
        status = integrated_app.get_status()
        print(f"📊 System Status:")
        print(f"   App: {status['app']}")
        print(f"   Model: {status['model']}")
        print(f"   Device: {status['device']}")
        print(f"   Gallery images: {status['gallery_images']:,}")
        print(f"   SPLICE: {'✅ Available' if status['splice_available'] else '❌ Unavailable'}")
        print(f"   QuARI: {'✅ Available' if status['quari_available'] else '❌ Unavailable'}")
        print(f"   Concepts: {status['concept_count']:,}")
        
        print(f"\n🌐 Starting server...")
        print(f"   Access at: http://localhost:5004")
        print(f"   Features:")
        print(f"   • 🎛️ Dynamic concept sliders with QuARI re-inference")
        print(f"   • 🔄 Gradient-based optimization on latent tokens")
        print(f"   • 🎨 SPLICE concept decomposition and reconstruction")
        print(f"   • ⚖️ Symmetric/asymmetric transformation modes")
        print(f"   • 📊 Real-time optimization monitoring")
        
        print(f"\n⚡ System Ready!")
        app.run(debug=True, host='0.0.0.0', port=5004)
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Please ensure PyTorch and dependencies are installed")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error starting app: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
