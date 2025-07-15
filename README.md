!pip install transformers
from transformers import pipeline

pipeline = pipeline("text-generation", model="catherinearnett/B-GPT_en_nl_simultaneous", device_map = "auto" )
pipeline(["Hi"])

#messing around and trying to use and upload transformers - text generation using the BLiMP model 
