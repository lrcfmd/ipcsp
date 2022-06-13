import os
from pathlib import Path
# Did you know that gulp can't read paths longer than 80 symbols?
# Yay, legacy
# I had issues like that around 2004, which didn't feel modern back then
# Looking positively into the future. Long string issues in 2040
# root_dir = Path(os.path.dirname(os.path.abspath(__file__)))
root_dir = Path(os.path.dirname(os.path.abspath(__file__)))
os.chdir(root_dir)
root_dir = Path('.')  # this bit doesn't acutually work, so os.path.join(".", ...)
