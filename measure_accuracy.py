import argparse
import json
from typing import Dict, List, Tuple
from difflib import SequenceMatcher

def load_json_file(filepath: str) -> Dict:
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_field_accuracy(ground_truth: str, extracted: str) -> float:

    gt = str(ground_truth).strip().lower()
    ext = str(extracted).strip().lower()

    if gt == "n/a" and ext == "n/a":
        return 1.0
    if gt == "n/a" or ext == "n/a":
        return 0.0

    if gt == ext:
        return 1.0

    similarity = SequenceMatcher(None, gt, ext).ratio()
    return similarity

def compare_documents(ground_truth_data: List[Dict], extracted_data: List[Dict]) -> Dict:

    # Create a mapping by filename
    gt_map = {doc['filename']: doc for doc in ground_truth_data}
    ext_map = {doc['filename']: doc for doc in extracted_data}

    fields_to_compare = [
        'first_name', 'last_name', 'address', 'state', 'country',
        'place_of_birth', 'document_type', 'document_number',
        'date_of_birth', 'document_issue_date', 'document_expiry_date',
        'class', 'sex', 'height', 'weight', 'hair', 'eyes'
    ]

    results = {
        'overall_accuracy': 0.0,
        'field_accuracies': {},
        'document_accuracies': {},
        'detailed_comparison': []
    }

    total_fields = 0
    total_accuracy = 0.0
    field_totals = {field: {'count': 0, 'accuracy': 0.0} for field in fields_to_compare}

    for filename in gt_map.keys():
        if filename not in ext_map:
            print(f"Warning: {filename} not found in extracted data")
            continue

        gt_doc = gt_map[filename]
        ext_doc = ext_map[filename]

        doc_comparison = {
            'filename': filename,
            'field_comparisons': [],
            'document_accuracy': 0.0
        }

        doc_total_accuracy = 0.0
        doc_field_count = 0

        for field in fields_to_compare:
            gt_value = gt_doc.get(field, 'N/A')
            ext_value = ext_doc.get(field, 'N/A')

            accuracy = calculate_field_accuracy(gt_value, ext_value)

            doc_comparison['field_comparisons'].append({
                'field': field,
                'ground_truth': gt_value,
                'extracted': ext_value,
                'accuracy': round(accuracy, 4),
                'match': accuracy == 1.0
            })

            # Update totals
            total_accuracy += accuracy
            total_fields += 1
            doc_total_accuracy += accuracy
            doc_field_count += 1

            # Update field-specific totals
            field_totals[field]['count'] += 1
            field_totals[field]['accuracy'] += accuracy

        doc_comparison['document_accuracy'] = round(doc_total_accuracy / doc_field_count * 100, 2)
        results['document_accuracies'][filename] = doc_comparison['document_accuracy']
        results['detailed_comparison'].append(doc_comparison)

    if total_fields > 0:
        results['overall_accuracy'] = round(total_accuracy / total_fields * 100, 2)

    for field, data in field_totals.items():
        if data['count'] > 0:
            results['field_accuracies'][field] = round(data['accuracy'] / data['count'] * 100, 2)

    return results

def print_accuracy_report(results: Dict):
    """Print a formatted accuracy report."""
    print("=" * 80)
    print("DOCUMENT EXTRACTION ACCURACY REPORT")
    print("=" * 80)
    print(f"\nOVERALL ACCURACY: {results['overall_accuracy']}%")
    print("\n" + "-" * 80)
    print("DOCUMENT-LEVEL ACCURACIES:")
    print("-" * 80)
    for filename, accuracy in results['document_accuracies'].items():
        print(f"  {filename:<30} {accuracy}%")

    print("\n" + "-" * 80)
    print("FIELD-LEVEL ACCURACIES:")
    print("-" * 80)
    for field, accuracy in sorted(results['field_accuracies'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {field:<25} {accuracy}%")

    print("\n" + "-" * 80)
    print("DETAILED FIELD COMPARISON:")
    print("-" * 80)

    for doc in results['detailed_comparison']:
        print(f"\n📄 {doc['filename']} (Accuracy: {doc['document_accuracy']}%)")
        print("  " + "-" * 76)

        for field_comp in doc['field_comparisons']:
            match_indicator = "✓" if field_comp['match'] else "✗"
            accuracy_pct = field_comp['accuracy'] * 100

            if not field_comp['match'] and field_comp['accuracy'] < 1.0:
                print(f"  {match_indicator} {field_comp['field']:<20} (Accuracy: {accuracy_pct:.1f}%)")
                print(f"    Ground Truth: {field_comp['ground_truth']}")
                print(f"    Extracted:    {field_comp['extracted']}")

def save_accuracy_report(results: Dict, output_path: str):
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n📊 Detailed accuracy report saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Measure accuracy of extracted document data against ground truth."
    )
    parser.add_argument(
        "--optimized",
        type=bool,
        default=False,
        help="Use extracted data from optimized images for comparison"
    )

    args = parser.parse_args()

    # File paths
    ground_truth_file = "results/ground_truth.json"
    extracted_data_file = "results/extracted_document_data.json" if not args.optimized else "results/extracted_document_data_optimized.json"
    output_file = "results/accuracy_report.json" if not args.optimized else "results/accuracy_report_optimized.json"

    # Load data
    ground_truth = load_json_file(ground_truth_file)
    extracted_data = load_json_file(extracted_data_file)

    # Compare documents
    print("Comparing documents...\n")
    results = compare_documents(
        ground_truth['ground_truth_data'],
        extracted_data['extracted_data']
    )

    # Print report
    print_accuracy_report(results)

    # Save report
    save_accuracy_report(results, output_file)

if __name__ == "__main__":
    main()
