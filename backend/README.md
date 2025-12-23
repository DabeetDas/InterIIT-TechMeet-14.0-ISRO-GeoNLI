# ISRO Satellite Image Analysis Backend

A comprehensive backend system for analyzing satellite imagery using Vision-Language Models (VLMs), YOLO object detection, and specialized models for captioning, grounding, and visual question answering (VQA).

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Environment Variables](#environment-variables)
- [Model Information](#model-information)
- [API Documentation](#api-documentation)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)

## Overview

This backend provides a REST API for processing satellite images with capabilities including:

- **Image Classification**: Classify images as SAR (Synthetic Aperture Radar), FCC (False Color Composite), or NCC (Natural Color Composite)
- **Caption Generation**: Generate detailed captions describing satellite imagery
- **Object Grounding**: Locate and return oriented bounding boxes (OBB) for objects in images
- **Visual Question Answering**: Answer binary, numeric, and semantic questions about images
- **Image Comparison**: Compare multiple images using VLM capabilities
- **Agriculture Analysis**: Specialized analysis for agricultural imagery

The system uses a two-server architecture:
1. **Modal FastAPI Server**: Hosts ML models on Modal's cloud platform
2. **Flask Backend Server**: Main API server that orchestrates requests and processes images

## Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│   Client/UI     │────────▶│  Flask Backend   │────────▶│  Modal Server   │
│                 │         │  (main.py)       │         │  (script.py)    │
│                 │         │                  │         │                 │
│                 │         │  - Request       │         │  - Qwen Models  │
│                 │         │    Processing    │         │  - Llama 3.1    │
│                 │         │  - Image Class.  │         │  - SAM Ensemble │
│                 │         │  - YOLO Models   │         │  - Grounding    │
│                 │         │  - Supabase      │         │  - VQA Models   │
└─────────────────┘         └──────────────────┘         └─────────────────┘
                                       │
                                       ▼
                            ┌──────────────────┐
                            │    Supabase      │
                            │  (Image Storage) │
                            └──────────────────┘
```

### Server Responsibilities

#### Modal FastAPI Server (`script.py`)
- Hosts all Vision-Language Models (VLMs) on GPU-enabled containers
- Provides inference endpoints for:
  - Base Qwen3-VL-8B model
  - Fine-tuned Qwen models (caption/VQA)
  - Grounding model with OBB detection
  - Llama 3.1 8B Instruct model
  - SAM2/SAM3 ensemble for mask generation
- Manages model caching and loading on Modal volumes
- Handles model quantization (4-bit) for efficient GPU usage

#### Flask Backend Server (`main.py`)
- Processes incoming API requests
- Classifies images (SAR/FCC/NCC) using YOLO models
- Handles image preprocessing and resizing
- Converts FCC to NCC when needed
- Orchestrates calls to Modal endpoints
- Manages temporary image storage via Supabase
- Formats responses according to VRSBench specification

## Prerequisites

Before setting up the backend, ensure you have:

1. **Python 3.8+** installed
2. **Modal account** with API token ([Get one here](https://modal.com))
3. **Supabase account** with:
   - Project URL
   - Service role key
   - Storage bucket named `images` (must be public)
4. **HuggingFace account** with access tokens (for gated models)
5. **YOLO model weights** (included in `./` directory):
   - `best.pt` - Image classification model
   - `runs_obb_train4_weights_last.pt` - OBB detection model
   - `last.pt` - Additional YOLO model
   - `yolo11x-obb.pt` - Base YOLO OBB model

## Setup Instructions

### Step 1: Setup Modal Server

The Modal server must be deployed first as the Flask backend depends on it.

1. **Install Modal CLI**:
   ```bash
   pip install modal
   ```

2. **Authenticate with Modal**:
   ```bash
   modal setup
   ```
   Follow the prompts to authenticate.

3. **Deploy the Modal Application**:
   ```bash
   cd backend
   modal deploy script.py
   ```

   This will:
   - Build Docker images with all dependencies
   - Download and cache model weights
   - Create Modal volumes for model storage
   - Deploy the FastAPI endpoints

   **Note**: First deployment may take approximately 6-7 minutes as it downloads all models.

4. **Get the Modal Endpoint URL**:
   After deployment, Modal will provide a URL like:
   ```
   https://yourusername--vlm-inference-server-fastapi-e-xxxxx-dev.modal.run
   ```

5. **Update Modal URL in Flask Backend**:
   Edit `modal_request.py` and update the `base_url`:
   ```python
   base_url = "https://your-actual-modal-url.modal.run"
   ```

### Step 2: Setup Flask Backend

1. **Navigate to the project directory**:
   ```bash
   cd backend
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create `.env` file**:
   Create a `.env` file in `./` directory:
   ```env
   SUPABASE_URL=your_supabase_project_url
   SUPABASE_KEY=your_supabase_service_role_key
   ```

5. **Verify YOLO model files**:
   Ensure the following model files are present in `./`:
   - `best.pt`
   - `runs_obb_train4_weights_last.pt`
   - `last.pt`
   - `yolo11x-obb.pt`

6. **Run the Flask server**:
   ```bash
   python main.py
   ```

   The server will start on `http://0.0.0.0:8000` (or the port specified in `main.py`).

### Step 3: Verify Installation

1. **Check Flask server health**:
   ```bash
   curl http://localhost:8000/health
   ```

   Expected response:
   ```json
   {
     "status": "ok"
   }
   ```

## Environment Variables

Create a `.env` file in the `./` directory with the following variables:

```env
# Supabase Configuration
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-service-role-key-here

# Optional: Modal Configuration (if using different endpoint)
MODAL_URL=https://your-modal-url.modal.run
```

**Important Notes**:
- `SUPABASE_KEY` should be the **service role key**, not the anon key
- Ensure the Supabase storage bucket `images` is created and set to **public**
- The Modal URL is configured in `modal_request.py` - update it after deploying Modal
- Also ensure that Create, Read, Update, Delete permissions are configured to true.

## Model Information

### Models Hosted on Modal

1. **Base Qwen3-VL-8B** (`unsloth/Qwen3-VL-8B-Instruct-bnb-4bit`)
   - Quantized to 4-bit for efficient inference
   - Used for general vision-language tasks

2. **Fine-tuned Qwen VQA** (`chhola14bhatoora/qwenvqa`)
   - LoRA adapter on base Qwen model
   - Specialized for caption and VQA tasks

3. **Grounding Model** (`chhola14bhatoora/qwengrounding`)
   - LoRA adapter for object grounding
   - Returns oriented bounding boxes (OBB) in format: `[cx, cy, w, h, angle]`

4. **Llama 3.1 8B Instruct** (`meta-llama/Llama-3.1-8B-Instruct`)
   - Text-only model for query enhancement
   - Used for SAR image analysis guidance

5. **SAM Ensemble**
   - **SAM2** (`AayushG06/SAM2.1-hiera-base-plus-finetuned-VRSBench`)
   - **SAM3** (gated repository)
   - **RemoteCLIP** (`chendelong/RemoteCLIP-ViT-B-32`)
   - Used for mask generation and refinement

### Local YOLO Models

Located in `./` directory:

- `best.pt`: Image classification (SAR/FCC/NCC)
- `runs_obb_train4_weights_last.pt`: Primary OBB detection
- `last.pt`: Secondary OBB detection
- `yolo11x-obb.pt`: Base OBB model

## API Documentation

### Flask Backend Endpoints

Base URL: `http://localhost:8000` (or your deployed URL)

#### 1. `/api` - Main VRSBench Pipeline

Processes a complete query JSON with caption, grounding, and attribute queries.

**Method**: `POST`

**Request Body**:
```json
{
  "input_image": {
    "image_id": "sample1.png",
    "image_url": "https://example.com/image.png",
    "metadata": {
      "width": 512,
      "height": 512,
      "spatial_resolution_m": 1.57
    }
  },
  "queries": {
    "caption_query": {
      "instruction": "Generate a detailed caption describing all visible elements in the satellite image."
    },
    "grounding_query": {
      "instruction": "Locate the swimming pool in the image."
    },
    "attribute_query": {
      "binary": {
        "instruction": "Is there any digit present in the bottom right corner?"
      },
      "numeric": {
        "instruction": "How many storage tanks are present in the scene?"
      },
      "semantic": {
        "instruction": "What is the color of the digit painted on the landing strip?"
      }
    }
  }
}
```

**Response**:
```json
{
  "input_image": {
    "image_id": "sample1.png",
    "image_url": "https://example.com/image.png",
    "metadata": {
      "width": 512,
      "height": 512,
      "spatial_resolution_m": 1.57
    }
  },
  "queries": {
    "caption_query": {
      "instruction": "...",
      "response": "The scene shows an satellite view of an airport runway..."
    },
    "grounding_query": {
      "instruction": "...",
      "response": [
        {
          "object-id": "1",
          "obbox": [
                        0.6290246324857288,
                        0.15450275032949543,
                        0.6862979976317555,
                        0.18559648444424137,
                        0.672192923514027,
                        0.21157744438218976,
                        0.6149195583680004,
                        0.1804837102674438
                    ]
        }
      ]
    },
    "attribute_query": {
      "binary": {
        "instruction": "...",
        "response": "Yes"
      },
      "numeric": {
        "instruction": "...",
        "response": 2.0
      },
      "semantic": {
        "instruction": "...",
        "response": "White"
      }
    }
  }
}
```

**cURL Example**:
```bash
curl -X POST http://localhost:8000/api \
  -H "Content-Type: application/json" \
  -d @sample1_query.json
```

#### 2. `/analyze` - UI Chat Interface

Endpoint for conversational interface with automatic query classification.

**Method**: `POST`

**Request Options**:
- **Query Parameter**: `?conversation=[JSON array]`
- **JSON Body**: `{"conversation": [...]}`

**Request Format**:
```json
[
  {
    "role": "user",
    "content": [
      {
        "type": "image",
        "image": "https://example.com/image.png"
      },
      {
        "type": "text",
        "text": "What objects are in this image?"
      },
      {
        "type": "dimensions",
        "height": 1.57
      }
    ]
  }
]
```

**Response**:
```json
{
  "answer": "The image contains two airplanes, storage tanks, and airport infrastructure."
}
```

For grounding queries, response includes:
```json
{
  "answer": "The bounding boxes are drawn in the image.",
  "groundingData": [
    {
      "swimming pool": [
        {
          "id": "1",
          "category": "swimming pool",
          "center": {"x": 256, "y": 256},
          "size": {"width": 100, "height": 50},
          "angle": 45
        }
      ]
    }
  ]
}
```

**Supported Query Types**:
- `captioning`: Generate image captions
- `grounding`: Object detection with OBB
- `vqa_float`: Numeric VQA (counts, measurements)
- `semantic`: Semantic VQA (colors, attributes)
- `binary`: Yes/No questions
- `general`: General VQA

#### 3. `/imageclass` - Image Classification

Classify an image as SAR, FCC, or NCC/Optical.

**Method**: `POST`

**Request Body**:
```json
{
  "image_url": "https://example.com/image.png"
}
```

**Response**:
```json
{
  "image_class": "SAR"
}
```

Possible values: `"SAR"`, `"FCC"`, `"Optical"` (NCC images are returned as "Optical")

#### 4. `/agriculture` - Agriculture Bot

Specialized endpoint for agriculture-related queries.

**Method**: `POST`

**Request**: Same format as `/analyze` endpoint

**Response**:
```json
{
  "answer": "Analysis result...",
  "graph": "optional graph data"
}
```

#### 5. `/compare` - Image Comparison

Compare multiple images using VLM capabilities.

**Method**: `POST`

**Request**: Conversation format with two images

#### 6. `/health` - Health Check

Check if the server is running.

**Method**: `GET`

**Response**:
```json
{
  "status": "ok"
}
```

### Modal FastAPI Endpoints

These endpoints are called internally by the Flask backend but can be accessed directly.

Base URL: Your Modal deployment URL (e.g., `https://username--app-name.modal.run`)

#### 1. `/qwen_base` - Base Qwen Model

**Method**: `POST`

**Query Parameter**: `messages` (JSON string)

**Example**:
```bash
curl -X POST "https://your-modal-url.modal.run/qwen_base?messages=[{\"role\":\"user\",\"content\":[...]}]"
```

#### 2. `/qwen_caption` - Caption Generation

**Method**: `POST`

Uses fine-tuned Qwen model for caption generation.

#### 3. `/qwen_grounding` - Object Grounding

**Method**: `POST`

Returns oriented bounding boxes in normalized format: `[cx, cy, w, h, angle]`

**Response Format**:
```json
{
  "response": [
    {
      "object-id": "1",
      "obbox": [0.5, 0.5, 0.1, 0.1, -45.0]
    }
  ]
}
```

#### 4. `/qwen_semantic_vqa` - Semantic VQA

**Method**: `POST`

Answers semantic questions about images (colors, attributes, etc.)

#### 5. `/llama3_3b` - Llama Model Inference

**Method**: `POST`

Text-only model for query enhancement and guidance generation.

#### 6. `/sam_ensemble_mask` - SAM Mask Generation

**Method**: `POST`

**Query Parameter**: `payload` (JSON string)

**Payload Format**:
```json
{
  "image_url": "https://example.com/image.png",
  "obb_box": [0.5, 0.5, 0.1, 0.1, 0.0],
  "text_query": "stadium"
}
```

**Response**:
```json
{
  "status": "success",
  "mask": "base64_encoded_mask",
  "mask_shape": [512, 512],
  "best_model": "SAM2",
  "best_score": 85.5,
  "image_shape": [512, 512]
}
```

## File Structure

```

├── backend/                           # Flask backend application
│   ├── main.py                        # Main Flask application
│   ├── modal_request.py               # Client for Modal endpoints
│   ├── caption.py                     # Caption generation module
│   ├── grounding.py                   # Object grounding module
│   ├── binary_vqa.py                  # Binary VQA module
│   ├── numeric_vqa.py                 # Numeric VQA module
│   ├── semantic_vqa.py                # Semantic VQA module
│   ├── general_vqa.py                 # General VQA module
│   ├── SAR_FCC.py                     # SAR/FCC processing
│   ├── fcc_to_ncc.py                  # FCC to NCC conversion
│   ├── classify_fcc_ncc.py            # Image classification
│   ├── overall_classifier.py          # Query type classification
│   ├── agriculture.py                 # Agriculture bot module
│   ├── best.pt                        # YOLO classification model
│   ├── runs_obb_train4_weights_last.pt # YOLO OBB model 1
│   ├── last.pt                        # YOLO OBB model 2
│   ├── yolo11x-obb.pt                 # Base YOLO OBB model
│   ├── sample1_query.json             # Example request
│   ├── sample1_response.json          # Example response
│   ├── sample2_query.json
│   ├── sample2_response.json
│   ├── sample3_query.json
│   ├── script.py                      # Modal deployment configuration for hosting ML models
│   └── .env                           # Environment variables (create this)
└── README.md                          # This file
```

### Key Files

- **`script.py`**: Modal deployment configuration for hosting ML models
- **`/main.py`**: Flask application with all API endpoints
- **`/modal_request.py`**: Client functions for calling Modal endpoints
- **`/*.py`**: Individual modules for different query types
- **`*.pt`**: YOLO model weights (required for local inference)

## Dependencies

### Python Packages

**Core Framework**:
- `flask` - Web framework
- `flask-cors` - CORS support
- `fastapi` - Modal server framework (installed in Modal image)

**Image Processing**:
- `pillow` (PIL) - Image manipulation
- `opencv-python-headless` - Computer vision operations
- `numpy` - Numerical operations

**ML Models**:
- `torch` - PyTorch (installed in Modal)
- `transformers` - HuggingFace transformers (Modal)
- `ultralytics` - YOLO models
- `peft` - Parameter-Efficient Fine-Tuning (Modal)
- `bitsandbytes` - Model quantization (Modal)

**Cloud Services**:
- `modal` - Modal platform SDK
- `supabase` - Supabase client for storage
- `requests` - HTTP client
- `huggingface_hub` - HuggingFace model downloads (Modal)

**Utilities**:
- `python-dotenv` - Environment variable management
- `qwen_vl_utils` - Qwen vision-language utilities (Modal)

### External Services

1. **Modal** - Cloud platform for hosting ML models
   - Requires Modal account and API token
   - Provides GPU instances (A100-80GB)

2. **Supabase** - Image storage and hosting
   - Requires project URL and service role key
   - Storage bucket named `images` (must be public)

3. **HuggingFace** - Model repository access
   - Requires access tokens for gated repositories
   - Models are automatically downloaded on first use

### Model Files

All YOLO model files (`.pt`) must be present in `./` directory:
- `best.pt` (~MB size)
- `runs_obb_train4_weights_last.pt` (~MB size)
- `last.pt` (~MB size)
- `yolo11x-obb.pt` (~MB size)

**Note**: These model files are not included in the repository and must be obtained separately.

## Usage Examples

### Example 1: Complete VRSBench Query

```bash
# Using sample1_query.json
curl -X POST http://localhost:8000/api \
  -H "Content-Type: application/json" \
  -d @./sample1_query.json
```

### Example 2: Image Classification

```bash
curl -X POST http://localhost:8000/imageclass \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/satellite_image.png"}'
```

### Example 3: Conversational Query (Caption)

```python
import requests
import json

conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://example.com/satellite_image.png"
            },
            {
                "type": "text",
                "text": "Describe this satellite image."
            }
        ]
    }
]

response = requests.post(
    "http://localhost:8000/analyze",
    params={"conversation": json.dumps(conversation)}
)

print(response.json())
```

### Example 4: Grounding Query

```python
conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://example.com/satellite_image.png"
            },
            {
                "type": "text",
                "text": "Locate all swimming pools in the image."
            }
        ]
    }
]

response = requests.post(
    "http://localhost:8000/analyze",
    params={"conversation": json.dumps(conversation)}
)

result = response.json()
print("Answer:", result["answer"])
if "groundingData" in result:
    print("Grounding Data:", result["groundingData"])
```

### Example 5: Numeric VQA

```python
conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://example.com/satellite_image.png"
            },
            {
                "type": "text",
                "text": "How many storage tanks are in this image?"
            },
            {
                "type": "dimensions",
                "height": 1.57  # Ground Sample Distance in meters
            }
        ]
    }
]

response = requests.post(
    "http://localhost:8000/analyze",
    params={"conversation": json.dumps(conversation)}
)

print(response.json())
```

## Troubleshooting

### Modal Server Issues

**Problem**: Modal deployment fails or models don't load

**Solutions**:
1. Check Modal authentication: `modal token show`
2. Verify HuggingFace tokens are set correctly in `script.py`
3. Check Modal volumes are created: `modal volume list`
4. Review Modal logs: `modal app logs vlm-inference-server`

**Problem**: Modal endpoint returns 404

**Solutions**:
1. Verify deployment succeeded: `modal app list`
2. Check the endpoint URL in `modal_request.py` matches your deployment
3. Ensure the Modal app is running (cold starts may take time)

### Flask Backend Issues

**Problem**: Import errors for local modules

**Solutions**:
1. Ensure you're running from `./` directory
2. Check Python path includes the current directory
3. Verify all Python files are present

**Problem**: YOLO model files not found

**Solutions**:
1. Verify all `.pt` files are in `./` directory
2. Check file permissions
3. Ensure files are not corrupted

**Problem**: Supabase connection errors

**Solutions**:
1. Verify `.env` file exists and has correct values
2. Check Supabase project is active
3. Ensure storage bucket `images` exists and is public
4. Verify service role key (not anon key) is used

**Problem**: Image download errors

**Solutions**:
1. Check internet connectivity
2. Verify image URLs are accessible
3. Ensure Supabase storage has write permissions
4. Check available disk space

### Performance Issues

**Problem**: Slow inference times

**Solutions**:
1. First request is slow due to Modal cold starts (6-7 minutes)
2. Subsequent requests should be faster (warm containers)
3. Consider increasing Modal `min_containers` setting
4. Check GPU availability in Modal

**Problem**: Memory errors

**Solutions**:
1. Reduce image size threshold in `reduce_image_size()` function
2. Clear Supabase storage of old temporary images
3. Monitor Modal container memory usage

### API Response Issues

**Problem**: Unexpected response format

**Solutions**:
1. Check request format matches API documentation
2. Verify image URLs are accessible
3. Review Flask logs for error messages
4. Ensure Modal endpoints return expected format

**Problem**: Grounding returns empty results

**Solutions**:
1. Verify query instruction is clear and specific
2. Check image quality and resolution
3. Ensure YOLO models are loaded correctly
4. Review grounding module logs

### General Tips

1. **Check Logs**: Both Flask and Modal provide detailed logs
   - Flask: Console output when running `python main.py`
   - Modal: Use `modal app logs vlm-inference-server`

2. **Test Modal Endpoints Directly**: Use curl or Postman to test Modal endpoints independently

3. **Verify Environment**: Use `/health` endpoint to verify Flask server is running

4. **Image Format**: Supported formats: PNG, JPEG, JPG
   - Images are automatically resized if > 750px
   - SAR images are processed with special handling
   - FCC images are converted to NCC for certain tasks

5. **Query Classification**: The system automatically classifies query types
   - Review `overall_classifier.py` for classification logic
   - Ensure queries are clear and specific for best results

## Additional Resources

- [Modal Documentation](https://modal.com/docs)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [YOLO Documentation](https://docs.ultralytics.com/)
- [Supabase Documentation](https://supabase.com/docs)
- [Qwen-VL Model Card](https://huggingface.co/Qwen/Qwen-VL)
