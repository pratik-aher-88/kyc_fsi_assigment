#!/usr/bin/env python3
"""
Utility script to check and print resolutions of images in documents and optimized_documents folders
"""

import os
from PIL import Image
from pathlib import Path


def get_image_resolution(image_path):
    """Get the resolution of an image"""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            dpi = img.info.get('dpi', ('N/A', 'N/A'))
            return width, height, dpi
    except Exception as e:
        return None, None, f"Error: {e}"


def print_resolutions_in_folder(folder_path, folder_name):
    """Print resolutions of all images in a folder"""
    print(f"\n{'='*80}")
    print(f"{folder_name.upper()}")
    print(f"{'='*80}")

    folder = Path(folder_path)

    if not folder.exists():
        print(f"❌ Folder '{folder_path}' does not exist!")
        return

    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}

    # Get all image files
    image_files = [f for f in folder.iterdir()
                   if f.is_file() and f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No images found in '{folder_path}'")
        return

    print(f"Found {len(image_files)} image(s)\n")

    # Print header
    print(f"{'Filename':<40} {'Resolution':<20} {'DPI':<20}")
    print(f"{'-'*40} {'-'*20} {'-'*20}")

    # Print details for each image
    for image_file in sorted(image_files):
        width, height, dpi = get_image_resolution(image_file)

        if width and height:
            resolution = f"{width} x {height}"
            dpi_str = f"{dpi[0]} x {dpi[1]}" if isinstance(dpi, tuple) else str(dpi)
        else:
            resolution = "N/A"
            dpi_str = str(dpi)

        # Truncate filename if too long
        filename = image_file.name
        if len(filename) > 38:
            filename = filename[:35] + "..."

        print(f"{filename:<40} {resolution:<20} {dpi_str:<20}")


def main():
    """Main function to check both folders"""
    # Get the project root directory (parent of utils folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Define folder paths
    documents_folder = project_root / "documents"
    optimized_folder = project_root / "optimized_documents"

    print("\n" + "="*80)
    print("IMAGE RESOLUTION CHECKER")
    print("="*80)

    # Check documents folder
    print_resolutions_in_folder(documents_folder, "Documents Folder")

    # Check optimized_documents folder
    print_resolutions_in_folder(optimized_folder, "Optimized Documents Folder")

    print("\n" + "="*80)
    print("DONE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
