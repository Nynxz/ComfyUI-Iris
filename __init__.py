"""ComfyUI-Iris: native Iris-3B (pixel-space text-to-image) support.

No custom nodes: importing the pack registers the model with ComfyUI's own loaders, so the
standard Load Diffusion Model / Load CLIP / Load VAE / KSampler graph runs it.
"""

from .iris.integration import register

register()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
