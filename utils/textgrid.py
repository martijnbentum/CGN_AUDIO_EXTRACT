from utils import handle_file
from utils import locations
import textgrids

def get_awd_textgrid_filename(filename):
    '''
    filename can be cgn_id e.g. fn00088
    '''
    f = handle_file.exists_locally(filename,locations.local_awd)
    if not f: f= handle_file.handle_original_cgn_awd_file(filename)
    return f
    

def load_awd_textgrid(filename):
    f = get_awd_textgrid_filename(filename)
    return textgrids.TextGrid(f)

def load_in_awd_textgrid(filename):
    filename = get_awd_textgrid_filename(filename)
    cgn_id = filename.split('/')[-1].split('.')[0]

