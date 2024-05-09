from typing import List, Dict, Any
        
class MarkDownWriter:
    def __init__(self, out_file: str):
        self.out_file = out_file
    
    def __enter__(self):
        self.file = open(self.out_file, 'w')
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()
    
    def write(self, text: str):
        self.file.write(text + '<br>\n')
        
    def write_header(self, header: str, level: int):
        self.file.write('#'*level + ' ' + header + '\n')
    
    def write_item(self, text: str):    
        self.file.write('- ' + text + '\n')
            
    def write_table(self, header: List[str], table: List[List[str]]):
        self.file.write('|'.join(header) + '\n')
        self.file.write('|'.join(['---']*len(header)) + '\n')
        for row in table:
            self.file.write('|'.join(row) + '\n')