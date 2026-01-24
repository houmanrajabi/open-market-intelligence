"""
Test script for Surya API Server

This script demonstrates how to use the Surya API for:
1. Layout detection
2. OCR text recognition
3. Full document processing (layout + OCR)

Usage:
    python scripts/test_surya_api.py --image path/to/image.jpg
    python scripts/test_surya_api.py --image path/to/image.jpg --api-url http://localhost:8002
"""

import argparse
import base64
import json
import requests
from pathlib import Path
from PIL import Image


def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string"""
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded


def test_health_check(api_url: str):
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("Testing Health Check")
    print("="*60)

    response = requests.get(f"{api_url}/health")

    if response.status_code == 200:
        print("✅ API is healthy")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"❌ Health check failed: {response.status_code}")
        print(response.text)


def test_list_models(api_url: str):
    """Test list models endpoint"""
    print("\n" + "="*60)
    print("Testing List Models")
    print("="*60)

    response = requests.get(f"{api_url}/v1/models")

    if response.status_code == 200:
        print("✅ Models retrieved successfully")
        data = response.json()
        for model in data.get("data", []):
            print(f"  • {model['id']} (owned by {model['owned_by']})")
    else:
        print(f"❌ Failed to list models: {response.status_code}")
        print(response.text)


def test_layout_detection(api_url: str, image_path: str):
    """Test layout detection endpoint"""
    print("\n" + "="*60)
    print("Testing Layout Detection")
    print("="*60)

    # Encode image
    print(f"Loading image: {image_path}")
    base64_image = encode_image_to_base64(image_path)

    # Prepare request
    payload = {
        "image": base64_image,
        "return_boxes": True,
        "return_polygons": False
    }

    print("Sending request to API...")
    response = requests.post(
        f"{api_url}/v1/layout/detect",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 200:
        print("✅ Layout detection successful")
        data = response.json()

        print(f"\nImage size: {data['image_size'][0]}x{data['image_size'][1]}")
        print(f"Detected {len(data['elements'])} elements:\n")

        for i, element in enumerate(data['elements'], 1):
            bbox = element['bbox']
            confidence = element.get('confidence', 'N/A')
            print(f"  {i}. {element['label']:15} | BBox: [{bbox[0]:4.0f}, {bbox[1]:4.0f}, {bbox[2]:4.0f}, {bbox[3]:4.0f}] | Confidence: {confidence}")

        return data
    else:
        print(f"❌ Layout detection failed: {response.status_code}")
        print(response.text)
        return None


def test_ocr(api_url: str, image_path: str):
    """Test OCR endpoint"""
    print("\n" + "="*60)
    print("Testing OCR Recognition")
    print("="*60)

    # Encode image
    print(f"Loading image: {image_path}")
    base64_image = encode_image_to_base64(image_path)

    # Prepare request
    payload = {
        "image": base64_image,
        "languages": ["en"]
    }

    print("Sending request to API...")
    response = requests.post(
        f"{api_url}/v1/ocr/recognize",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 200:
        print("✅ OCR successful")
        data = response.json()

        print(f"\nImage size: {data['image_size'][0]}x{data['image_size'][1]}")
        print(f"Recognized {len(data['text_lines'])} text lines\n")

        print("Full Text:")
        print("-" * 60)
        print(data['full_text'])
        print("-" * 60)

        return data
    else:
        print(f"❌ OCR failed: {response.status_code}")
        print(response.text)
        return None


def test_document_processing(api_url: str, image_path: str):
    """Test full document processing endpoint"""
    print("\n" + "="*60)
    print("Testing Full Document Processing (Layout + OCR)")
    print("="*60)

    # Encode image
    print(f"Loading image: {image_path}")
    base64_image = encode_image_to_base64(image_path)

    # Prepare request
    payload = {
        "image": base64_image,
        "languages": ["en"],
        "return_polygons": False
    }

    print("Sending request to API...")
    response = requests.post(
        f"{api_url}/v1/document/process",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 200:
        print("✅ Document processing successful")
        data = response.json()

        print(f"\nImage size: {data['image_size'][0]}x{data['image_size'][1]}")
        print(f"Processed {len(data['elements'])} elements:\n")

        for element in data['elements']:
            bbox = element['bbox']
            text_preview = element.get('text', '[No text]')[:50]
            if len(element.get('text', '')) > 50:
                text_preview += "..."

            print(f"  [{element['element_id']}] {element['type']:10} | BBox: [{bbox[0]:4.0f}, {bbox[1]:4.0f}, {bbox[2]:4.0f}, {bbox[3]:4.0f}]")
            if element.get('text'):
                print(f"      Text: {text_preview}")

        return data
    else:
        print(f"❌ Document processing failed: {response.status_code}")
        print(response.text)
        return None


def test_file_upload(api_url: str, image_path: str):
    """Test file upload endpoint for layout detection"""
    print("\n" + "="*60)
    print("Testing File Upload (Layout Detection)")
    print("="*60)

    print(f"Loading image: {image_path}")

    with open(image_path, 'rb') as f:
        files = {'file': (Path(image_path).name, f, 'image/jpeg')}

        print("Uploading file to API...")
        response = requests.post(
            f"{api_url}/v1/layout/detect/upload",
            files=files
        )

    if response.status_code == 200:
        print("✅ File upload successful")
        data = response.json()

        print(f"\nDetected {len(data['elements'])} elements via upload")
        return data
    else:
        print(f"❌ File upload failed: {response.status_code}")
        print(response.text)
        return None


def main():
    parser = argparse.ArgumentParser(description="Test Surya API Server")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to test image"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8002",
        help="Surya API base URL (default: http://localhost:8002)"
    )
    parser.add_argument(
        "--skip-health",
        action="store_true",
        help="Skip health check"
    )
    parser.add_argument(
        "--skip-layout",
        action="store_true",
        help="Skip layout detection test"
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Skip OCR test"
    )
    parser.add_argument(
        "--skip-document",
        action="store_true",
        help="Skip document processing test"
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip file upload test"
    )

    args = parser.parse_args()

    # Validate image path
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        return

    print("="*60)
    print("  Surya API Test Suite")
    print("="*60)
    print(f"API URL: {args.api_url}")
    print(f"Test Image: {args.image}")

    # Run tests
    try:
        if not args.skip_health:
            test_health_check(args.api_url)

        test_list_models(args.api_url)

        if not args.skip_layout:
            test_layout_detection(args.api_url, args.image)

        if not args.skip_ocr:
            test_ocr(args.api_url, args.image)

        if not args.skip_document:
            test_document_processing(args.api_url, args.image)

        if not args.skip_upload:
            test_file_upload(args.api_url, args.image)

        print("\n" + "="*60)
        print("  ✅ All tests completed!")
        print("="*60)

    except requests.exceptions.ConnectionError:
        print(f"\n❌ Error: Could not connect to {args.api_url}")
        print("\nMake sure:")
        print("  1. Surya API is running on VastAI")
        print("  2. SSH tunnel is established:")
        print(f"     ssh -L 8002:localhost:8002 -p <port> root@<vastai-host> -N")
        print("  3. The API URL is correct")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
