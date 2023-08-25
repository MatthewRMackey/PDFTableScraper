from os import listdir
from os.path import isfile, join
import time
import csv

import openai
from PyPDF2 import PdfReader, PdfWriter
import tiktoken

def getTokens(s):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(s))

def getTranscribedPdfs():
    with open("./config/transcribed.csv", "r+") as fr:
        transcribed = list(csv.reader(fr, delimiter=","))
    
    if len(transcribed) > 0:
        return transcribed[0]
    else:
        return []

def addTranscribed(path):
    processed_pdfs = getTranscribedPdfs()
    if path[path.rindex("/")+1:] not in processed_pdfs:
        processed_pdfs.append(path[path.rindex("/")+1:])
        with open("./config/transcribed.csv", "w+", newline="") as fw:
            writer = csv.writer(fw)
            writer.writerow(processed_pdfs)

def getTrimmedPdfs():
    trim_pdf_path = "./data/trimmed_pdfs/"
    return [join(trim_pdf_path, pdf) for pdf in listdir(trim_pdf_path) if isfile(join(trim_pdf_path, pdf))]

def trancribeTables(path):
    reader = PdfReader(path)
    token_limit = 4096
    with open("./config/table_prompt.txt", "r") as fr:
        system_prompt = "".join(fr.readlines())

    responses = {}
    for index, page in enumerate(reader.pages):
        contents = page.extract_text()
        total_tokens = getTokens(system_prompt) + getTokens(contents)

        if total_tokens < token_limit*.85:
            try:
                response = openai.ChatCompletion.create(model="gpt-4", messages=[{"role":"system", "content":system_prompt}, {"role":"user", "content":contents}])
                responses[index] = response["choices"][0]["message"]["content"]
            except:
                responses[index] = "APIErrorResponseError"
        else:
            responses[index] = "TooManyTokensError"
        time.sleep(3)
    
    with open("./data/raw_tables/"+path[path.rindex("/")+1:-4]+".txt", "w+", encoding="utf-8") as fw:
        for page in responses:
            fw.write("\nTrimmed Page "+str(page)+"\n\n"+responses[page]+"\n")
       
def TableProcessor():
    with open("./config/key.txt", "r") as fr:
        openai.api_key = fr.readline()

    transcribed = getTranscribedPdfs()
    pdf_paths = getTrimmedPdfs()
    start_time = time.time()
    for path in pdf_paths:
        pdf_name = path[path.rindex("/")+1:]
        if pdf_name not in transcribed:
            print("Processing: "+pdf_name)
            trancribeTables(path)
            addTranscribed(path)
            print("Total runtime:", time.time()-start_time)
            time.sleep(20)


if __name__ == "__main__":
    TableProcessor()
