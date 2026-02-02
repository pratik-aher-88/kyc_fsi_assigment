# Document Text Extraction System

## Overview

AI-powered OCR system that extracts structured data from identity documents (passports, driver's licenses, ID cards) using Fireworks AI's vision model with built-in encryption.

## Quick Start

### Standard Mode
```bash
# Place documents in documents/ folder
python process_and_extract_text.py
```

### Optimized Mode (Recommended - 30-40% cost savings)
```bash
python optimize_images_for_ocr.py  # Optimize images first
python process_and_extract_text.py --optimized
```

## Key Features

- **Multi-document support** with batch processing (up to 15 images)
- **Built-in encryption** for PII protection
- **60-80% smaller files** in optimized mode
- **Structured JSON output** with validation

## Setup

```bash
# Install packages
pip install fireworks-ai pillow cryptography

# Set environment variables
export FIREWORKS_API_KEY="your_api_key"  # Get from fireworks.ai
export ENCRYPTION_KEY="your_password"    # For encrypting results
```

## Command-Line Arguments

| Argument | Description |
|----------|-------------|
| `--optimized` | Use preprocessed images from `optimized_documents/` |
| `--no-encrypt` | Disable encryption (not recommended) |
| `--decrypt FILE` | Decrypt an encrypted results file |
| `--output FILE` | Custom output path (use with `--decrypt`) |

```bash
# Basic usage
python process_and_extract_text.py --optimized

# Decrypt results
python process_and_extract_text.py --decrypt results/data.json.encrypted
```

## Extracted Data Fields

**Personal**: first_name, last_name, date_of_birth, sex, place_of_birth
**Physical**: height, weight, hair, eyes
**Address**: address, state, country
**Document**: document_type, document_number, issue_date, expiry_date, class, filename

## Output Format

Results saved as:
- Standard: `results/extracted_document_data.json.encrypted`
- Optimized: `results/extracted_document_data_optimized.json.encrypted`

Includes extracted data array + metadata (tokens, duration, averages)

## Security & Encryption

**Encryption is enabled by default** using Fernet encryption to protect PII data.

```bash
# Setup
export ENCRYPTION_KEY="YourPassword"

# Decrypt results
python process_and_extract_text.py --decrypt results/data.json.encrypted
```

⚠️ Use `--no-encrypt` only for testing with non-sensitive data.

## Why Use Optimized Mode?

| Benefit | Improvement |
|---------|-------------|
| File Size | 60-80% smaller |
| Speed | 20-30% faster |
| Cost | 30-40% lower |
| Accuracy | Better (enhanced contrast) |

**When to use:** Large batches (10+ docs), cost-sensitive workflows, or poor quality images

## Configuration

- **Model**: Qwen3-VL-235B (temperature: 0.5)
- **Batch size**: 15 images max per batch
- **Formats**: .png, .jpg, .jpeg

## Performance

Tracks images processed, duration, token usage, and cost metrics. Check [fireworks.ai/pricing](https://fireworks.ai/pricing) for current rates.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Missing API key | `export FIREWORKS_API_KEY="your_key"` |
| Missing encryption key | `export ENCRYPTION_KEY="your_password"` or use `--no-encrypt` |
| No images found | Check `documents/` folder exists with .png/.jpg/.jpeg files |
| Invalid JSON response | Use optimized mode, verify image quality, retry |
| Inaccurate extraction | Use optimized mode, ensure high resolution (>800px) |
| Slow processing | Use optimized mode, check connection, smaller batches |
| High costs | Always use optimized mode (saves 30-40%) |

## Related Files

- [optimize_images_for_ocr.py](optimize_images_for_ocr.py) - Image preprocessing
- [OPTIMIZE_IMAGES_README.md](OPTIMIZE_IMAGES_README.md) - Optimization docs
- [ENCRYPTION_GUIDE.md](ENCRYPTION_GUIDE.md) - Security guide

## Best Practices

1. **Use optimized mode** for production (run `optimize_images_for_ocr.py` first)
2. **Enable encryption** for PII data (default behavior)
3. **Monitor token usage** in metadata to track costs
4. **Validate critical fields** (dates, document numbers)
5. **Backup encrypted results** regularly
6. **Use .gitignore** for .env and *.encrypted files

## Security Notes

- Never commit API keys or encryption keys
- Generate strong keys: `openssl rand -base64 32`
- Delete decrypted files after use
- Use secure file permissions: `chmod 600 results.json`
- Ensure GDPR compliance for document processing

---

**Tip**: For optimal results and cost savings, always run `optimize_images_for_ocr.py` before using `--optimized`!
