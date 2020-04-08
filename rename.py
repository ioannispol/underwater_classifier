import os
"""
Renames the filenames within the same directory to be Unix friendly
(1) Changes spaces to hyphens
(2) Makes lowercase
Usage:
python rename.py
"""

path =  os.getcwd()
filenames = os.listdir(path)

for filename in filenames:
    os.rename(filename, filename.replace(" ", " img").lower())
    
