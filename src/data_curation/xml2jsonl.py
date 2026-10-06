import xml.etree.ElementTree as ET
import json
import argparse

def xml_to_jsonl(xml_file, jsonl_file, target_tag='DescriptorRecord'):
    with open(xml_file, 'r', encoding='utf-8') as infile, \
         open(jsonl_file, 'w', encoding='utf-8') as outfile:
        
        for event, elem in ET.iterparse(infile, events=('end',)):
            if elem.tag == target_tag:
                # Convert to dictionary
                record = element_to_dict(elem)
                # Write to JSONL file
                outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
                # Clear processed elements to save memory
                elem.clear()

def element_to_dict(element):
    """Convert XML element to dictionary"""
    result = {}
    
    # Handle attributes
    if element.attrib:
        result['@attributes'] = element.attrib
    
    # Handle child elements
    for child in element:
        child_data = element_to_dict(child)
        
        # If child has text content, add to result
        if child.text and child.text.strip():
            result[child.tag] = child.text.strip()
        
        # If there's child element data, merge to result
        if child_data:
            # Handle duplicate elements (convert to list)
            if child.tag in result:
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data
    
    return result

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("xml_file", help="XML file path")
    parser.add_argument("jsonl_file", help="Output JSONL file path")
    parser.add_argument("--tag", default="DescriptorRecord", help="Target XML tag name, default is DescriptorRecord")
    args = parser.parse_args()

    xml_to_jsonl(args.xml_file, args.jsonl_file, target_tag=args.tag)
    print(f"Converted {args.xml_file} to {args.jsonl_file}")
