import os
import pandas as pd
import numpy as np

# Convert ToTTo table into WebNLG Format
def get_headers(table):
  headers = []
  row_idx=0
  
  # For every header row
  while str(table[row_idx][0]['is_header']).lower()=='true':
    row_values = []
    
    # Iterate through the columns and collect the values
    for item in table[row_idx]:
      row_values.extend([item['value']]*item['column_span'])
    row_idx += 1
    headers.append(row_values)
  
  # Concatenate the header names (Needed if table is multilevel)
  headers = list(zip(*headers))
  headers = [' '.join(h) for h in headers]
  return headers, row_idx

def get_body(table, start_row=0):
  body = []
  
  # For each row in the body
  for row in table[start_row:]:
    row_values = []
    
    # Collect the values in the row
    for item in row:
      row_values.extend([item['value']]*item['column_span'])
    body.append(row_values)
  return body

def match_head_to_body(headers, body):
  
  # Generate a list of (<Col Name>,<Value>) tuples per row
  output_temp = [list(zip(headers,row)) for row in body]
  
  # Turn into WebNLG format 
  # i.e. <Col1 Name>|||<Value1>\t<Col2 Name>|||<Value2>
  output_final = []
  for row in output_temp:
    s = "<tab>".join(f"{value[0]}|||{value[1]}" for value in row)
    output_final.append(s)

  # Concatenate all rows together
  output_final = '<tab>'.join(output_final)
  
  return output_final


def parse_table_totto_webnlg(path):
    """
    Input: List of lists, corresponding to the ToTTo format
    Output: Single string corresponding to WebNLG formatted table
    """
    # Convert ToTTo to WebNLG format
    headers, start_row = get_headers(table)
    body = get_body(table, start_row)
    output = match_head_to_body(headers, body)
    return output

def parse_table_logicnlg_webnlg(path):
    lst_files = [i for i in os.listdir(path) if '.csv' in i]
    output_files = {}
    for filename in lst_files:
        df = pd.read_csv(path+'/'+filename, delimiter='#')
        output = match_head_to_body(headers=list(df.columns), 
                                    body=np.array(df))
        output_files[filename] = output
    return output_files
    
# Write to file
# textfile = open("sample_table.txt", "w")
# for element in output:
#     textfile.write(element + "\n")
# textfile.close()
