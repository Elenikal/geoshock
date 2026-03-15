# sitecustomize.py — suppress all warnings at Python startup
# Place in project root; activated via PYTHONPATH=. streamlit run dashboard/app.py
import warnings
import logging
import os
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
logging.basicConfig(level=logging.ERROR)
for name in list(logging.Logger.manager.loggerDict):
    logging.getLogger(name).setLevel(logging.ERROR)
