import os
import torch
import numpy as np
import folder_paths
from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModelForCausalLM

# Register model_path
models_dir = os.path.join(folder_paths.base_path, "models")
ovis_dir = os.path.join(models_dir, "ovis")
os.makedirs(ovis_dir, exist_ok=True)

# Add Ovis models folder to folder_paths
if "ovis" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["ovis"] = ([ovis_dir], folder_paths.supported_pt_extensions)

class Ovis2ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["AIDC-AI/Ovis2-34B", "AIDC-AI/Ovis2-16B", "AIDC-AI/Ovis2-8B", 
                               "AIDC-AI/Ovis2-2B", "AIDC-AI/Ovis2-1B"], {"default": "AIDC-AI/Ovis2-8B"}),
                "precision": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "max_token_length": ("INT", {"default": 32768, "min": 2048, "max": 65536}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "auto_download": (["enable", "disable"], {"default": "enable"}),
            }
        }
    
    RETURN_TYPES = ("OVIS2_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "Ovis2"

    def download_model(self, model_name):
        """Download the model files from Hugging Face if they don't exist locally."""
        local_dir = os.path.join(ovis_dir, model_name.split('/')[-1])
        
        print(f"Downloading Ovis2 model: {model_name} to {local_dir}")
        try:
            # Create a complete snapshot of the repository locally
            snapshot_download(
                repo_id=model_name,
                local_dir=local_dir,
                local_dir_use_symlinks=False  # Use actual files instead of symlinks for better compatibility
            )
            print(f"Successfully downloaded {model_name} to {local_dir}")
            return local_dir
        except Exception as e:
            print(f"Error downloading model: {str(e)}")
            raise RuntimeError(f"Failed to download model {model_name}. Error: {str(e)}")

    def check_model_files(self, model_name):
        """Check if the model files already exist locally."""
        local_dir = os.path.join(ovis_dir, model_name.split('/')[-1])
        
        # Check for config.json as a basic indicator that the model exists
        config_path = os.path.join(local_dir, "config.json")
        return os.path.exists(config_path), local_dir

    def load_model(self, model_name, precision, max_token_length, device, auto_download):
        print(f"Loading Ovis2 model: {model_name}")
        
        # Set precision
        if precision == "bfloat16":
            dtype = torch.bfloat16
        elif precision == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        # Check if model exists locally
        model_exists, local_dir = self.check_model_files(model_name)
        
        # Download model if it doesn't exist and auto_download is enabled
        if not model_exists and auto_download == "enable":
            self.download_model(model_name)

        # Load the model
        try:
            # First try loading from local directory
            if model_exists or auto_download == "enable":
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        local_dir,
                        torch_dtype=dtype,
                        multimodal_max_length=max_token_length,
                        trust_remote_code=True
                    ).to(device)
                except Exception as e:
                    print(f"Error loading from local directory, falling back to HuggingFace: {str(e)}")
                    # Fall back to loading directly from HuggingFace
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=dtype,
                        multimodal_max_length=max_token_length,
                        trust_remote_code=True
                    ).to(device)
            else:
                # Load directly from HuggingFace
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    multimodal_max_length=max_token_length,
                    trust_remote_code=True
                ).to(device)
            
            # Get tokenizers
            text_tokenizer = model.get_text_tokenizer()
            visual_tokenizer = model.get_visual_tokenizer()
            
            return ({"model": model, "text_tokenizer": text_tokenizer, "visual_tokenizer": visual_tokenizer},)
        except Exception as e:
            print(f"Error loading Ovis2 model: {str(e)}")
            raise e


class Ovis2ImageCaption:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OVIS2_MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "Describe this image in detail.", "multiline": True}),
                "max_new_tokens": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "generate_caption"
    CATEGORY = "Ovis2"

    def generate_caption(self, model, image, prompt, max_new_tokens, temperature):
        model_data = model
        model = model_data["model"]
        text_tokenizer = model_data["text_tokenizer"]
        
        # Convert ComfyUI image to PIL image format
        # ComfyUI images are in format [batch, height, width, channel]
        if len(image.shape) == 4:
            # Handle batch of images - take the first one
            image_tensor = image[0]
        else:
            # Handle single image
            image_tensor = image
            
        pil_image = Image.fromarray((image_tensor * 255).astype(np.uint8))
        
        # Process the image with Ovis2
        query = f"<image>\n{prompt}"
        
        try:
            # Preprocess inputs
            _, input_ids, pixel_values = model.preprocess_inputs(query, [pil_image], max_partition=9)
            
            # Generate response
            outputs = model.generate(
                input_ids=input_ids.to(model.device),
                pixel_values=pixel_values.to(model.device),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True
            )
            
            # Decode the generated text
            generated_text = text_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the generated response (remove the instruction)
            response = generated_text[len(query):]
            
            return (response,)
        except Exception as e:
            print(f"Error generating caption: {str(e)}")
            return (f"Error generating caption: {str(e)}",)


class Ovis2MultiImageInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OVIS2_MODEL",),
                "image1": ("IMAGE",),
                "prompt": ("STRING", {"default": "Analyze these images. What are the similarities and differences?", "multiline": True}),
                "max_new_tokens": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.1}),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "process_multi_images"
    CATEGORY = "Ovis2"

    def process_multi_images(self, model, image1, prompt, max_new_tokens, temperature, image2=None, image3=None, image4=None):
        model_data = model
        model = model_data["model"]
        text_tokenizer = model_data["text_tokenizer"]
        
        # Convert ComfyUI images to PIL image format
        images = []
        
        # Process first image
        if len(image1.shape) == 4:
            # Handle batch of images - take the first one
            pil_image1 = Image.fromarray((image1[0] * 255).astype(np.uint8))
        else:
            # Handle single image
            pil_image1 = Image.fromarray((image1 * 255).astype(np.uint8))
            
        images.append(pil_image1)
        
        # Add optional additional images if provided
        if image2 is not None:
            if len(image2.shape) == 4:
                pil_image2 = Image.fromarray((image2[0] * 255).astype(np.uint8))
            else:
                pil_image2 = Image.fromarray((image2 * 255).astype(np.uint8))
            images.append(pil_image2)
        
        if image3 is not None:
            if len(image3.shape) == 4:
                pil_image3 = Image.fromarray((image3[0] * 255).astype(np.uint8))
            else:
                pil_image3 = Image.fromarray((image3 * 255).astype(np.uint8))
            images.append(pil_image3)
        
        if image4 is not None:
            if len(image4.shape) == 4:
                pil_image4 = Image.fromarray((image4[0] * 255).astype(np.uint8))
            else:
                pil_image4 = Image.fromarray((image4 * 255).astype(np.uint8))
            images.append(pil_image4)
        
        # Process the images with Ovis2
        query = f"<image>\n{prompt}"
        
        try:
            # Preprocess inputs
            _, input_ids, pixel_values = model.preprocess_inputs(query, images, max_partition=9)
            
            # Generate response
            outputs = model.generate(
                input_ids=input_ids.to(model.device),
                pixel_values=pixel_values.to(model.device),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True
            )
            
            # Decode the generated text
            generated_text = text_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the generated response (remove the instruction)
            response = generated_text[len(query):]
            
            return (response,)
        except Exception as e:
            print(f"Error processing images: {str(e)}")
            return (f"Error processing images: {str(e)}",)


class Ovis2VideoFramesDescription:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OVIS2_MODEL",),
                "frames": ("IMAGE",),
                "prompt": ("STRING", {"default": "Describe what's happening in this video.", "multiline": True}),
                "max_new_tokens": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.1}),
                "frame_skip": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "max_frames": ("INT", {"default": 16, "min": 1, "max": 32, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "describe_video"
    CATEGORY = "Ovis2"

    def describe_video(self, model, frames, prompt, max_new_tokens, temperature, frame_skip, max_frames):
        model_data = model
        model = model_data["model"]
        text_tokenizer = model_data["text_tokenizer"]
        
        # Convert ComfyUI image frames to PIL images
        pil_images = []
        
        # Check if frames is a single image or a batch
        if len(frames.shape) == 4 and frames.shape[0] > 1:
            # It's a batch of images
            # Apply frame skipping and limit to max_frames
            selected_indices = list(range(0, min(frames.shape[0], max_frames * frame_skip), frame_skip))
            
            for i in selected_indices:
                if i < frames.shape[0]:
                    pil_image = Image.fromarray((frames[i] * 255).astype(np.uint8))
                    pil_images.append(pil_image)
        else:
            # It's a single image, treat it as a 1-frame video
            if len(frames.shape) == 4:
                pil_image = Image.fromarray((frames[0] * 255).astype(np.uint8))
            else:
                pil_image = Image.fromarray((frames * 255).astype(np.uint8))
            pil_images.append(pil_image)
        
        # Process the images with Ovis2
        query = f"<image>\n{prompt}"
        
        try:
            # Preprocess inputs
            _, input_ids, pixel_values = model.preprocess_inputs(query, pil_images, max_partition=9)
            
            # Generate response
            outputs = model.generate(
                input_ids=input_ids.to(model.device),
                pixel_values=pixel_values.to(model.device),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True
            )
            
            # Decode the generated text
            generated_text = text_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the generated response (remove the instruction)
            response = generated_text[len(query):]
            
            return (response,)
        except Exception as e:
            print(f"Error describing video: {str(e)}")
            return (f"Error describing video: {str(e)}",)


# Register the nodes
NODE_CLASS_MAPPINGS = {
    "Ovis2ModelLoader": Ovis2ModelLoader,
    "Ovis2ImageCaption": Ovis2ImageCaption,
    "Ovis2MultiImageInput": Ovis2MultiImageInput,
    "Ovis2VideoFramesDescription": Ovis2VideoFramesDescription,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Ovis2ModelLoader": "Load Ovis2 Model",
    "Ovis2ImageCaption": "Ovis2 Image Caption",
    "Ovis2MultiImageInput": "Ovis2 Multi-Image Analysis",
    "Ovis2VideoFramesDescription": "Ovis2 Video Frames Description",
}
