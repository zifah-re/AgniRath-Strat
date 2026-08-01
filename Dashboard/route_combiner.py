import xml.etree.ElementTree as ET
from pathlib import Path

def analyze_kml_structure(kml_bytes: bytes):
    root = ET.fromstring(kml_bytes)
    
    # Strip namespaces
    for el in root.iter():
        if '}' in el.tag:
            el.tag = el.tag.split('}', 1)[1]
            
    # Determine depth dynamically matching your logic
    depth = 1
    folder = True
    while folder:
        try:
            tag = root.find('Document/' + ('Folder/' * depth) * 1 + 'name')
            if tag is not None and len(tag.text):
                depth += 1
            else:
                depth -= 1
                folder = False
        except:
            depth -= 1
            folder = False

    # Extract all folders and their child placemarks
    available_folders = []
    folder_elements = root.findall('Document' + ('/Folder') * depth)
    
    for f_idx, f_el in enumerate(folder_elements):
        f_name = f_el.find('name')
        f_text = f_name.text if f_name is not None else f"Folder {f_idx + 1}"
        
        placemarks = []
        placemark_elements = f_el.findall('Placemark')
        for p_idx, p_el in enumerate(placemark_elements):
            p_name = p_el.find('name')
            p_text = p_name.text if p_name is not None else f"Placemark {p_idx + 1}"
            
            # Verify it contains line coordinates before listing
            if p_el.find("LineString/coordinates") is not None or p_el.find("Polygon/outerBoundaryIs/LinearRing/coordinates") is not None:
                if p_el.find("LineString/coordinates") is not None:
                    coords=p_el.find("LineString/coordinates").text
                else:
                    coords=p_el.find("Polygon/outerBoundaryIs/LinearRing/coordinates").text
                placemarks.append({"index": p_idx, "name": p_text, "coords": coords})
                
        available_folders.append({
            "index": f_idx,
            "name": f_text,
            "placemarks": placemarks
        })
        
    return root, available_folders
def main(kml_bytes):
    _,structural_opts=analyze_kml_structure(kml_bytes)
    for x,i in enumerate(structural_opts):
        print(f"{x+1}. {i["name"]}")
    ch=int(input("Enter choice: "))
    for x,i in enumerate(structural_opts[ch-1]["placemarks"]):
        print(f"{x+1}. {i["name"]}")
    ch1=input("Enter order of combination (eg. 1,3,2,4): ").strip().split(',')
    order=[(int(i)-1,1) if '*' not in i else (int(i.strip("*"))-1,-1) for i in ch1]
    coords=" ".join([" ".join(structural_opts[ch-1]["placemarks"][i]["coords"].strip().split()[:-1][::j] if i!=order[-1][0] else structural_opts[ch-1]["placemarks"][i]["coords"].strip().split()[::j]) for i,j in order])
    return coords

if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    file_name="2026 Sasol Solar Challenge Route (Publish).kml" #input("Enter file name: ")
    FILE_PATH=SCRIPT_DIR / file_name
    file=open(FILE_PATH,'r')
    data=file.read()
    print(main(kml_bytes=data))
    
        
