# Encryption at Rest - Usage Guide

## Overview
The KYC processor now encrypts extracted PII data at rest using Fernet symmetric encryption (AES 128-bit in CBC mode).

## Setup

### 1. Install Required Package
```bash
pip install cryptography
```

### 2. Set Encryption Key
Set a strong encryption key as an environment variable:

```bash
# Linux/Mac
export ENCRYPTION_KEY="your-strong-password-here-min-32-chars"

# Windows
set ENCRYPTION_KEY=your-strong-password-here-min-32-chars
```

**IMPORTANT**:
- Use a strong, unique password (32+ characters recommended)
- Store this key securely (password manager, vault, or secrets manager)
- If you lose the key, encrypted data CANNOT be recovered

## Usage

### Processing Documents (with Encryption - Default)
```bash
python process_and_extract_text.py
```

This will:
1. Extract data from documents
2. Write results **directly** to encrypted file
3. Create `extracted_document_data.json.encrypted`
4. **No unencrypted file is created** (PII never touches disk in plain text)

### Processing Without Encryption (Not Recommended)
```bash
python process_and_extract_text.py --no-encrypt
```

This creates an unencrypted JSON file instead.

### Decrypting Results
```bash
# Decrypt to original filename (removes .encrypted extension)
python process_and_extract_text.py --decrypt results/extracted_document_data.json.encrypted

# Decrypt to custom output path
python process_and_extract_text.py --decrypt results/extracted_document_data.json.encrypted --output decrypted_data.json
```

## Security Best Practices

1. **Key Management**
   - Never hardcode encryption keys in your code
   - Use environment variables or a secrets manager (AWS Secrets Manager, HashiCorp Vault, etc.)
   - Rotate encryption keys periodically

2. **Access Control**
   - Limit file system permissions on encrypted files
   - Use `chmod 600` on Linux/Mac to restrict access

3. **Secure Key Storage**
   ```bash
   # Store in a secure file with restricted permissions
   echo "ENCRYPTION_KEY=your-key-here" > .env
   chmod 600 .env
   source .env
   ```

4. **Production Deployment**
   - Use cloud-based key management services (AWS KMS, Azure Key Vault, GCP KMS)
   - Enable audit logging for all decryption operations
   - Implement key rotation policies

## Python API Usage

```python
from process_and_extract_text import KYCProcessor

# With encryption (default - recommended)
processor = KYCProcessor(
    api_key="your_api_key",
    encrypt=True,  # Default
    encryption_key="your-encryption-key"  # Optional, defaults to ENCRYPTION_KEY env var
)
processor.process()

# Without encryption (plain text)
processor = KYCProcessor(
    api_key="your_api_key",
    encrypt=False
)
processor.process()

# Decrypt a file programmatically
KYCProcessor.decrypt_file(
    encrypted_file_path="results/extracted_document_data.json.encrypted",
    encryption_key="your-encryption-key",
    output_path="decrypted_data.json"
)
```

## File Structure

### Default (Encrypted)
```
results/
└── extracted_document_data.json.encrypted    # Encrypted file only
```

### With --no-encrypt Flag
```
results/
└── extracted_document_data.json              # Unencrypted file only
```

## Troubleshooting

### Error: "Encryption is enabled but no key provided"
- Set the `ENCRYPTION_KEY` environment variable
- Or pass `encryption_key` parameter to KYCProcessor

### Error: "Invalid token" during decryption
- You're using the wrong encryption key
- The file may be corrupted
- Ensure you're using the same key that was used for encryption

### Want to save unencrypted file instead?
```bash
python process_and_extract_text.py --no-encrypt
```
This will create a plain text JSON file instead of encrypted.

## Technical Details

- **Algorithm**: Fernet (AES 128-bit CBC + HMAC)
- **Key Derivation**: SHA-256 hash of password
- **Encoding**: Base64 URL-safe encoding
- **Library**: `cryptography` (Python)
