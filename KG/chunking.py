import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=600,
  chunk_overlap=200,
  length_function=len,
  is_separator_regex=False,
)

def split_data_from_file(file_path):
  with open(file_path, 'r', encoding='utf-8') as f:
    medicine_list = json.load(f)
  
  chunks_with_metadata = []
  
  for medicine in medicine_list:
    medicine_name = medicine.get('Medicine Name', 'UnknownMedicine')
    generics_indicated = medicine.get('Generics indicated', '')
    therapeutic_class = medicine.get('Therapeutic Class', '')
    indications = medicine.get('Indications', '')
    manufacturer = medicine.get('Manufacturer', '')

    # Convert all medicine details to a single text string
    medicine_text = f"Medicine Name: {medicine_name}\n" # Explicitly state field names
    medicine_text += f"Generics indicated: {generics_indicated}\n"
    medicine_text += f"Therapeutic Class: {therapeutic_class}\n"
    medicine_text += f"Indications: {indications}\n"
    medicine_text += f"Indications Details: {medicine.get('Indications Details', '')}\n"
    medicine_text += f"Weight (mg): {medicine.get('Weight (mg)', '')}\n"
    medicine_text += f"Weight (ml/other): {medicine.get('Weight (ml/other)', '')}\n"
    
    forms = []
    if medicine.get('Tablet') == 1:
      forms.append("Tablet")
    if medicine.get('Syrup') == 1:
      forms.append("Syrup")
    if medicine.get('Ointment') == 1:
      forms.append("Ointment")
    if medicine.get('Drop') == 1:
      forms.append("Drop")
    if medicine.get('Injection') == 1:
      forms.append("Injection")
    medicine_text += f"Available Forms: {', '.join(forms) if forms else 'N/A'}\n"
    medicine_text += f"Manufacturer: {manufacturer}"
    
    medicine_chunks = text_splitter.split_text(medicine_text)
    
    chunk_seq_id = 0
    for chunk_text in medicine_chunks:
      chunks_with_metadata.append({
        "text": chunk_text,
        "medicine_name_ref": medicine_name, # Key used to link to the Medicine node
        "chunkSeqId": chunk_seq_id,
        "chunkId": f"medicine-{medicine_name.replace(' ', '_').lower()}-chunk{chunk_seq_id:04d}",
        "source": "medicine_database"
      })
      chunk_seq_id += 1
      
    print(f"Processed {medicine_name} - Split into {chunk_seq_id} chunks")
      
  return chunks_with_metadata
