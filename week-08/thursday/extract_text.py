import zipfile
import xml.etree.ElementTree as ET

def get_docx_text(path):
    """
    Take the path of a docx file as argument, return the text in html.
    """
    document = zipfile.ZipFile(path)
    xml_content = document.read('word/document.xml')
    document.close()
    tree = ET.fromstring(xml_content)

    return ' '.join([node.text for node in tree.iter() if node.text])

if __name__ == "__main__":
    print(get_docx_text('W8_Thursday_DailyAssignment.docx'))
