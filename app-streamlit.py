import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import base64
import uuid

import transformers
from datasets import Dataset,load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer


st.set_page_config(
    page_title="Named Entity Recognition Tagger", page_icon="📘"
)

######### Streamlit-related functions #########

# def download_button(object_to_download, download_filename, button_text):
#     """
#     Generates a link to download the given object_to_download.
#     From: https://discuss.streamlit.io/t/a-download-button-with-custom-css/4220
#     Params:
#     ------
#     object_to_download:  The object to be downloaded.
#     download_filename (str): filename and extension of file. e.g. mydata.csv,
#     some_txt_output.txt download_link_text (str): Text to display for download
#     link.
#     button_text (str): Text to display on download button (e.g. 'click here to download file')
#     pickle_it (bool): If True, pickle file.
#     Returns:
#     -------
#     (str): the anchor tag to download object_to_download
#     Examples:
#     --------
#     download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
#     download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')
#     """

#     if isinstance(object_to_download, bytes):
#         pass

#     elif isinstance(object_to_download, pd.DataFrame):
#         print('here')
#         object_to_download = object_to_download.to_csv(index=False)
#     # Try JSON encode for everything else
#     else:
#         object_to_download = json.dumps(object_to_download)

#     try:
#         # some strings <-> bytes conversions necessary here
#         b64 = base64.b64encode(object_to_download.encode()).decode()
#         print(b64)
#     except AttributeError as e:
#         b64 = base64.b64encode(object_to_download).decode()

#     button_uuid = str(uuid.uuid4()).replace("-", "")
#     button_id = re.sub("\d+", "", button_uuid)

#     custom_css = f""" 
#         <style>
#             #{button_id} {{
#                 display: inline-flex;
#                 align-items: center;
#                 justify-content: center;
#                 background-color: rgb(255, 255, 255);
#                 color: rgb(38, 39, 48);
#                 padding: .25rem .75rem;
#                 position: relative;
#                 text-decoration: none;
#                 border-radius: 4px;
#                 border-width: 1px;
#                 border-style: solid;
#                 border-color: rgb(230, 234, 241);
#                 border-image: initial;
#             }} 
#             #{button_id}:hover {{
#                 border-color: rgb(246, 51, 102);
#                 color: rgb(246, 51, 102);
#             }}
#             #{button_id}:active {{
#                 box-shadow: none;
#                 background-color: rgb(246, 51, 102);
#                 color: white;
#                 }}
#         </style> """

#     dl_link = (
#         custom_css
#         + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br><br>'
#     )

#     st.markdown(dl_link, unsafe_allow_html=True)


#@st.cache
def convert_df(df:pd.DataFrame):
     return df.to_csv(index=False).encode('utf-8')

#@st.cache
def convert_json(df:pd.DataFrame):
    result = df.to_json(orient="index")
    parsed = json.loads(result)
    json_string = json.dumps(parsed)
    #st.json(json_string, expanded=True)
    return json_string

st.title("📘 Named Entity Recognition Tagger")


######### App-related functions #########

@st.cache(allow_output_mutation=True)
def load_model():

    model = AutoModelForTokenClassification.from_pretrained("NER_roberta_updated/model/")
    trainer = Trainer(model=model)

    tokenizer = AutoTokenizer.from_pretrained("NER_roberta_updated/tokenizer/")

    return trainer, model, tokenizer


id2tag={0: 'O',
        1: 'B-corporation',
        2: 'I-corporation',
        3: 'B-creative-work',
        4: 'I-creative-work',
        5: 'B-group',
        6: 'I-group',
        7: 'B-location',
        8: 'I-location',
        9: 'B-person',
        10: 'I-person',
        11: 'B-product',
        12: 'I-product'}

def tag_sentence(text:str):
      # convert our text to a tokenized sequence
      inputs = tokenizer(text, truncation=True, return_tensors="pt")
      # get outputs
      outputs = model(**inputs)
      # convert to probabilities with softmax
      probs = outputs[0][0].softmax(1)
      # get the tags with the highest probability
      word_tags = [(tokenizer.decode(inputs['input_ids'][0][i].item()), id2tag[tagid.item()], np.round(probs[i][tagid].item() *100,2) ) 
                    for i, tagid in enumerate (probs.argmax(axis=1))]

      df=pd.DataFrame(word_tags, columns=['word', 'tag', 'probability'])
      return df

with st.form(key='my_form'):

    x1 = st.text_input(label='Enter a sentence:', max_chars=250)
    submit_button = st.form_submit_button(label='🏷️ Create tags')

if submit_button:
    if re.sub('\s+','',x1)=='':
        st.error('Please enter a non-empty sentence.')

    elif re.match(r'\A\s*\w+\s*\Z', x1):
        st.error("Please enter a sentence with at least one word")
    
    else:
        st.markdown("### Tagged Sentence")
        st.header("")

        Trainer, model, tokenizer = load_model()  
        results=tag_sentence(x1)
        
        cs, c1, c2, c3, cLast = st.columns([0.75, 1.5, 1.5, 1.5, 0.75])

        with c1:
            #csvbutton = download_button(results, "results.csv", "📥 Download .csv")
            csvbutton = st.download_button(label="📥 Download .csv", data=convert_df(results), file_name= "results.csv", mime='text/csv', key='csv')
        with c2:
            #textbutton = download_button(results, "results.txt", "📥 Download .txt")
            textbutton = st.download_button(label="📥 Download .txt", data=convert_df(results), file_name= "results.text", mime='text/plain',  key='text')
        with c3:
            #jsonbutton = download_button(results, "results.json", "📥 Download .json")
            jsonbutton = st.download_button(label="📥 Download .json", data=convert_json(results), file_name= "results.json", mime='application/json',  key='json')

        st.header("")
        
        c1, c2, c3 = st.columns([1, 3, 1])
        
        with c2:

             st.table(results.style.background_gradient(subset=['probability']).format(precision=2))

st.header("")
st.header("")
st.header("")
with st.expander("ℹ️ - About this app", expanded=True):


    st.write(
        """     
-   The **Named Entity Recognition Tagger** app is a tool that performs named entity recognition.
-   The available entitites are: *corporation*, *creative-work*, *group*, *location*, *person* and *product*.
-   The app uses the [RoBERTa model](https://huggingface.co/roberta-large), fine-tuned on the [wnut](https://huggingface.co/datasets/wnut_17) dataset.      
-   The model uses the **byte-level BPE tokenizer**. Each sentece is first tokenized.
-   For more info regarding the data science part, check this [post](https://towardsdatascience.com/named-entity-recognition-with-deep-learning-bert-the-essential-guide-274c6965e2d?sk=c3c3699e329e45a8ed93d286ae04ef10).      
       """
    )

