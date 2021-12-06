import convert
import argparse
from tqdm.auto import tqdm


parser = argparse.ArgumentParser(description='''Create dataset from text file. Ensure that the text file contains newline delimited sentences either in the target language for
adaptation, or one of the constituent languages of the code-mixed language *except* the base language''')

parser.add_argument('--input', '-i', type=str,
                    help='Input file', required=True)
parser.add_argument('--output', '-o', type=str,
                    help='Output file as CSV', required=True)
parser.add_argument('--target', '-t', type=str,
                    help='Language code of one of the constituent languages of the code-mixed language except the base language', required=True)
parser.add_argument('--base', '-b', type=str,
                    help='Base language code used to originally pre-train Transformer', required=True)
parser.add_argument('--format', '-f', type=bool, help='Input data format is code-mixed', required=True)
args = parser.parse_args()
print(args)

input_file_path = args.input
output_file_path = args.output
target_language = args.target
base_language = args.base
format = args.format
num_failed = 0

with open(input_file_path, encoding='utf8') as input_file, open(output_file_path, 'w', encoding='utf8') as output_file:
    output_file.write(f"translation,transliteration\n")
    lines = input_file.readlines()
    for line in tqdm(lines):
        line = line.strip()
        converted_line_details = convert.getConvertedText(
            line, base_language, target_language)
        translation = converted_line_details['translation'].strip()
        if format:
            transliteration = line
        else:
            transliteration = converted_line_details['transliteration'].strip()
        output_file.write(f"{translation},{transliteration}\n")

print(f"Conversion completed. Failed conversions: {num_failed}")
