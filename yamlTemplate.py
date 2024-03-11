#!/usr/bin/env python
import os
import sys


def generate_yaml(user_defined_name, path_to_dataset, output_file="aurum/ddprofiler/Dataset.yml"):
    print(user_defined_name, path_to_dataset)
    yaml_content = f"""
#######
# This template is from Aurum https://github.com/mitdbg/aurum-datadiscovery/blob/master/ddprofiler/src/main/resources/template.yml
# This file is specified as input to the ddprofiler, which uses it to extract a list of
# sources that are necessary to process and profile.
#######    
api_version: 0

# In sources we include as many data sources as desired
sources:

# Include a source for each data source to configure

  # name to identify each source, e.g., migration_database, production, lake, etc.
- name: {user_defined_name}
  # indicate the type of source, one of [csv, postgres, mysql, oracle10g, oracle11g]
  type: csv
  config:
    # path indicates where the CSV files live
    path: {path_to_dataset}
    # separator used in the CSV files
    separator: ','
"""
    with open(output_file, 'w') as file:
        file.write(yaml_content)
if __name__ == "__main__":
    print( len(sys.argv))
    if len(sys.argv) <3:
        print("Please provide the defined dataset name and the path to the dataset")
    if len(sys.argv) >4 :
        print("Please only provide the defined dataset name, the path to the dataset and the output file")
    user_defined_name = sys.argv[1]
    path_to_dataset = sys.argv[2]

    if len(sys.argv) == 4:
        output_file = sys.argv[3]
        generate_yaml(user_defined_name, path_to_dataset, output_file)

    if len(sys.argv) == 3:
        generate_yaml(user_defined_name, path_to_dataset)