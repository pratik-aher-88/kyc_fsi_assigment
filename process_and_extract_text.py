import argparse
import base64
import json
import os
import time
from typing import Dict, List
from fireworks import Fireworks
from PIL import Image
import io
from cryptography.fernet import Fernet
import hashlib



class KYCProcessor:
    def __init__(self, api_key: str, optimized: bool = False, encrypt: bool = True, encryption_key: str = None):
        self.api_key = api_key
        self.model = "accounts/fireworks/models/qwen3-vl-235b-a22b-instruct"
        self.temperature = 0.5
        self.images_folder = "documents" if not optimized else "optimized_documents"
        self.results_file = "results/extracted_document_data.json" if not optimized else "results/extracted_document_data_optimized.json"
        self.system_prompt = self._system_prompt()
        self.encrypt = encrypt
        self.encryption_key = self._get_encryption_key(encryption_key) if encrypt else None

        self.client = Fireworks(api_key=self.api_key)
    
    def _system_prompt(self):
        return (
            "To extract text from identity documents (passports, driver's licenses, ID cards)."
            "Your task is to carefully read and extract ONLY the printed text from the document fields listed below. "
            "CRITICAL: Return ONLY a valid JSON array of objects. Do NOT include any markdown formatting, headers, explanations, or code blocks. "
            "The response must start with '[' and end with ']' and be valid JSON that can be parsed directly.\n\n"

            "CRITICAL INSTRUCTIONS:\n"
            "- Extract text EXACTLY as it appears on the document. Do not guess, infer, or modify.\n"
            "- Pay special attention to numbers and measurements - transcribe them character-by-character.\n"
            "- For HEIGHT: Look very carefully at the printed value. Common formats include:\n"
            "  * '5-05' (means 5 feet 5 inches)\n"
            "  * '5\'8\"' (means 5 feet 8 inches)\n"
            "- For WEIGHT: Extract the exact number with unit (e.g., '125 lb', '180 lbs', '165').\n"
            "- For DATES: Extract in the exact format shown on the document.\n"
            "- For STATE: Use the full state name, not abbreviations (e.g., 'California' not 'CA').\n"
            "- Do NOT provide physical descriptions of people in photos.\n"
            "- If a field is not present or not legible, use 'N/A'.\n\n"

            "Fields to extract:\n"
            "- first_name: Person's first/given name\n"
            "- last_name: Person's last/family name\n"
            "- address: Full street address if present\n"
            "- state: Full state name (not abbreviated)\n"
            "- country: Country name or code as shown\n"
            "- place_of_birth: Birthplace as printed\n"
            "- document_type: Type of document (e.g., 'driver license', 'passport', 'ID card')\n"
            "- document_number: The document's unique identifier\n"
            "- date_of_birth: Birth date in the format shown\n"
            "- document_issue_date: When the document was issued\n"
            "- document_expiry_date: The document's expiration date (typically labeled 'Exp' or 'Expiration Date'). CRITICAL: Exercise extreme care when reading this field, as certain digits are easily confused: '0' vs '8', '3' vs '5', '1' vs '7', '6' vs '8'. Note: Driver's licenses typically expire on the same month and day after a fixed number of years, so reference the issue date if uncertain.\n"
            "- class: Driver's license class if applicable (e.g., 'C', 'B', 'A')\n"
            "- sex: Gender as printed (usually 'M' or 'F'). In passports, in some cases, its printed below 'Sex' section e.g. 'M', 'F' \n"
            "- height: Extract EXACTLY as printed - DO NOT interpret or convert\n"
            "- weight: Weight with unit if shown\n. Some licenses have stopped putting weight so if you are not certain of weight field put 'N/A'"
            "- hair: Hair color code as printed (e.g., 'BRN', 'BLK')\n"
            "- eyes: Eye color code as printed (e.g., 'BRN', 'BLU')\n"
            "- filename: The image filename provided\n\n"

            "JSON Schema (all keys must be in snake_case):\n"
            "{\n"
            "  \"first_name\": \"<string>\",\n"
            "  \"last_name\": \"<string>\",\n"
            "  \"address\": \"<string>\",\n"
            "  \"state\": \"<string>\",\n"
            "  \"country\": \"<string>\",\n"
            "  \"place_of_birth\": \"<string>\",\n"
            "  \"filename\": \"<string>\",\n"
            "  \"document_type\": \"<string>\",\n"
            "  \"document_number\": \"<string>\",\n"
            "  \"date_of_birth\": \"<string>\",\n"
            "  \"document_issue_date\": \"<string>\",\n"
            "  \"document_expiry_date\": \"<string>\",\n"
            "  \"class\": \"<string>\",\n"
            "  \"sex\": \"<string>\",\n"
            "  \"height\": \"<string>\",\n"
            "  \"weight\": \"<string>\",\n"
            "  \"hair\": \"<string>\",\n"
            "  \"eyes\": \"<string>\"\n"
            "}\n\n"

            "IMPORTANT: Double-check numbers and measurements. Extract text character-by-character without interpretation. "
            "Accuracy is critical - if you're unsure about a character, look more carefully at the image."
        )

    def _get_encryption_key(self, key: str = None) -> bytes:
        if key is None:
            key = os.getenv("ENCRYPTION_KEY")
            if key is None:
                raise ValueError(
                    "Encryption is enabled but no key provided. "
                    "Set ENCRYPTION_KEY environment variable or pass encryption_key parameter."
                )

        # Generate a valid Fernet key from the password using SHA256
        key_bytes = hashlib.sha256(key.encode()).digest()
        return base64.urlsafe_b64encode(key_bytes)

    def _write_encrypted(self, data: bytes, file_path: str):
        fernet = Fernet(self.encryption_key)
        encrypted_data = fernet.encrypt(data)

        encrypted_file_path = file_path + '.encrypted'
        with open(encrypted_file_path, 'wb') as f:
            f.write(encrypted_data)

        return encrypted_file_path

    @staticmethod
    def decrypt_file(encrypted_file_path: str, encryption_key: str, output_path: str = None):

        key_bytes = hashlib.sha256(encryption_key.encode()).digest()
        fernet_key = base64.urlsafe_b64encode(key_bytes)
        fernet = Fernet(fernet_key)

        with open(encrypted_file_path, 'rb') as f:
            encrypted_data = f.read()

        decrypted_data = fernet.decrypt(encrypted_data)

        if output_path is None:
            output_path = encrypted_file_path.replace('.encrypted', '')

        with open(output_path, 'wb') as f:
            f.write(decrypted_data)

        print(f"✓ File decrypted successfully: {output_path}")
        return output_path

    def _create_payload(self, image_content):
        messages = [
            {
            "role": "system",
            "content": self.system_prompt
            },
            {
            "role": "user",
            "content": image_content
            }
        ]
        
        return {
            "model": self.model,
            "max_tokens": 4096,
            "top_p": 0.5,
            "top_k": 50,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "temperature": self.temperature,
            "messages": messages
        }

    def _send_request(self, payload: Dict):

        self.start_time = time.time()

        response = self.client.chat.completions.create(
            model=payload["model"],
            messages=payload["messages"],
            max_tokens=payload.get("max_tokens", 4096),
            top_p=payload.get("top_p", 1),
            top_k=payload.get("top_k", 100),
            presence_penalty=payload.get("presence_penalty", 0),
            frequency_penalty=payload.get("frequency_penalty", 0),
            temperature=payload.get("temperature", 0)
        )

        self.end_time = time.time()

        return response
    
    def _get_image_files(self) -> List[str]:

        image_files = [
            f for f in os.listdir(self.images_folder) 
            if f.lower().endswith(("png", "jpg", "jpeg"))
        ]
        
        if not image_files:
            raise FileNotFoundError(f"No valid images found in the '{self.images_folder}' folder.")
        
        return image_files

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as img_f:
            return base64.b64encode(img_f.read()).decode("utf-8")
    
    def _process_image_content(self, image_files: List[str]) -> List[Dict]:

        image_content = []

        for image_file in image_files:

            image_path = os.path.join(self.images_folder, image_file)
            encoded_image = self._encode_image(image_path)

            mime_type = "image/png" if image_file.lower().endswith("png") else "image/jpeg"

            image_content.extend([
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{encoded_image}"
                    }
                },
                {
                    "type": "text",
                    "text": f"This image is named {image_file}."
                }
            ])

        return image_content
    
    def _metrics_calculation(self, num_images: int):

        duration_ms = (self.end_time - self.start_time) * 1000
        average_ms_per_image = duration_ms / num_images
        return duration_ms, average_ms_per_image

    def _extract_usage_info(self, response_dict: Dict):

        usage = response_dict.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        return prompt_tokens, total_tokens

    def _write_results(self, extracted_data: List[Dict], num_images: int,
                      duration_ms: float, average_ms: float, prompt_tokens: int = None, total_tokens: int = None):
        # Use passed in token counts if provided (for batching), otherwise use instance variables
        prompt_tokens = prompt_tokens
        total_tokens = total_tokens

        results = {
            "extracted_data": extracted_data,
            "metadata": {
                "num_images": num_images,
                "duration_ms": round(duration_ms, 3),
                "average_ms_per_image": round(average_ms, 3),
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens,
                "average_tokens_per_image": total_tokens//num_images if num_images > 0 else 0,
            }
        }

        if self.encrypt:
            # Write directly to encrypted file
            json_bytes = json.dumps(results, indent=4).encode('utf-8')
            encrypted_path = self._write_encrypted(json_bytes, self.results_file)
            print(f"✓ Results encrypted: {encrypted_path}")
            self.results_file = encrypted_path
        else:
            # Write unencrypted file
            with open(self.results_file, 'w') as f:
                json.dump(results, f, indent=4)

    def _print_statistics(self, num_images: int, duration_ms: float, average_ms: float, prompt_tokens: int = None, total_tokens: int = None):
        prompt_tokens = prompt_tokens
        total_tokens = total_tokens

        print(f"Processing Stats:")
        print(f"{'-'*40}")
        print(f"Images processed: {num_images}")
        print(f"Total duration: {duration_ms:.3f} ms ({duration_ms/1000:.3f} seconds)")
        print(f"Average per image: {average_ms:.3f} ms ({average_ms/1000:.3f} seconds)")
        print(f"Prompt tokens: {prompt_tokens}")
        print(f"Total tokens: {total_tokens}")
        print(f"Average tokens per image: ({total_tokens//num_images if num_images > 0 else 0} tokens)")
        print(f"{'-'*40}")

    def _clean_json_response(self, content_str: str) -> str:
        """
        Clean JSON response by removing markdown code block formatting.
        Handles cases where response starts with ```json and ends with ```
        """
        content_str = content_str.strip()

        # Remove opening markdown code block
        if content_str.startswith("```json"):
            content_str = content_str[7:]  # Remove ```json
        elif content_str.startswith("```"):
            content_str = content_str[3:]  # Remove ```

        # Remove closing markdown code block
        if content_str.endswith("```"):
            content_str = content_str[:-3]

        return content_str.strip()


    def _process_response(self, response, num_images: int, duration_ms: float, average_ms: float):

        if not response.choices or len(response.choices) == 0:
            print("No valid content found in the response.")
            return None

        content_str = response.choices[0].message.content

        cleaned_content = self._clean_json_response(content_str)

        try:
            extracted_data = json.loads(cleaned_content)

            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response.__dict__
            prompt_tokens, total_tokens = self._extract_usage_info(response_dict)

            self._write_results(extracted_data, num_images,
                                duration_ms, average_ms, prompt_tokens, total_tokens)

            print("\nExtracted text saved to:", self.results_file)
            self._print_statistics(num_images, duration_ms, average_ms,  prompt_tokens, total_tokens)

            return extracted_data

        except json.JSONDecodeError as e:
            print(f"The assistant response was not valid JSON.")
            print(f"Error: {e}")
            return None

    def _process_batch(self, batch_files: List[str], batch_num: int, total_batches: int):
        print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch_files)} images)...")

        image_content = self._process_image_content(batch_files)
        payload = self._create_payload(image_content)

        response = self._send_request(payload)

        if not response.choices or len(response.choices) == 0:
            print(f"No valid content found in the response for batch {batch_num}.")
            return None, 0, 0

        content_str = response.choices[0].message.content
        cleaned_content = self._clean_json_response(content_str)

        try:
            extracted_data = json.loads(cleaned_content)
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response.__dict__
            usage = response_dict.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)


            print("Intermediate batch results",prompt_tokens, total_tokens)

            return extracted_data, prompt_tokens, total_tokens

        except json.JSONDecodeError as e:
            print(f"Batch {batch_num} response was not valid JSON. Error: {e}")
            return None, 0, 0

    def process(self):
        image_files = self._get_image_files()
        num_of_images = len(image_files)

        print(f"Found {num_of_images} images to process in '{self.images_folder}' folder.")

        MAX_BATCH_SIZE = 20 # max size is 30. To be safe we are putting a threshold at 20.

        # If 20 or fewer images, process normally
        if num_of_images <= MAX_BATCH_SIZE:
            print(f"Processing all {num_of_images} images in a single batch.")

            image_content = self._process_image_content(image_files)
            payload = self._create_payload(image_content)
            response = self._send_request(payload)
            duration_ms, average_ms = self._metrics_calculation(num_of_images)
            self._process_response(response, num_of_images, duration_ms, average_ms)

        else:
            # Batch processing for more than 20 images
            num_batches = (num_of_images + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE
            print(f"Processing in {num_batches} batches of up to {MAX_BATCH_SIZE} images each.")

            all_extracted_data = []
            total_prompt_tokens = 0
            total_tokens_sum = 0
            overall_start_time = time.time()

            for i in range(num_batches):
                start_idx = i * MAX_BATCH_SIZE
                end_idx = min((i + 1) * MAX_BATCH_SIZE, num_of_images)
                batch_files = image_files[start_idx:end_idx]

                batch_data, prompt_tokens, total_tokens = self._process_batch(
                    batch_files, i + 1, num_batches
                )

                if batch_data:
                    # Handle both list and single dict responses
                    if isinstance(batch_data, list):
                        all_extracted_data.extend(batch_data)
                    else:
                        all_extracted_data.append(batch_data)

                    total_prompt_tokens += prompt_tokens
                    total_tokens_sum += total_tokens

            overall_end_time = time.time()
            overall_duration_ms = (overall_end_time - overall_start_time) * 1000
            overall_average_ms = overall_duration_ms / num_of_images if num_of_images > 0 else 0

            # Write combined results
            self._write_results(
                all_extracted_data,
                num_of_images,
                overall_duration_ms,
                overall_average_ms,
                total_prompt_tokens,
                total_tokens_sum
            )

            print("\n" + "="*40)
            print("ALL BATCHES COMPLETED")
            print("="*40)
            print(f"\nExtracted text saved to: {self.results_file}")
            self._print_statistics(
                num_of_images,
                overall_duration_ms,
                overall_average_ms,
                total_prompt_tokens,
                total_tokens_sum
            )



def main():
    parser = argparse.ArgumentParser(
        description="Extract information from documents in documents folder."
    )
    parser.add_argument(
        "--optimized",
        action="store_true",
        help="Use pre processed images optimized for OCR"
    )
    parser.add_argument(
        "--no-encrypt",
        action="store_true",
        help="Disable encryption and save as plain text (not recommended for PII)"
    )
    parser.add_argument(
        "--decrypt",
        type=str,
        metavar="FILE",
        help="Decrypt an encrypted results file (provide path to .encrypted file)"
    )
    parser.add_argument(
        "--output",
        type=str,
        metavar="FILE",
        help="Output path for decrypted file (only used with --decrypt)"
    )

    args = parser.parse_args()

    # Handle decryption mode
    if args.decrypt:
        encryption_key = os.getenv("ENCRYPTION_KEY")
        if not encryption_key:
            raise ValueError("ENCRYPTION_KEY environment variable is not set")

        KYCProcessor.decrypt_file(args.decrypt, encryption_key, args.output)
        return

    # Normal processing mode
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        raise ValueError("FIREWORKS_API_KEY environment variable is not set")

    encrypt = not args.no_encrypt

    if encrypt:
        print("🔒 Encryption ENABLED - writing directly to encrypted file")
    else:
        print("⚠️  WARNING: Encryption DISABLED - PII will be stored in plain text")

    kycpreprocessor = KYCProcessor(api_key, args.optimized, encrypt=encrypt)
    kycpreprocessor.process()


if __name__ == "__main__":
    main()