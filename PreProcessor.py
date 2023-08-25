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

def getClassifiedPdfs():
    with open("./config/classified.csv", "r+") as fr:
        classified = list(csv.reader(fr, delimiter=","))
    
    if len(classified) > 0:
        return classified[0]
    else:
        return []

def addClassified(path):
    processed_pdfs = getClassifiedPdfs()
    if path[path.rindex("/")+1:] not in processed_pdfs:
        processed_pdfs.append(path[path.rindex("/")+1:])
        with open("./config/classified.csv", "w+", newline="") as fw:
            writer = csv.writer(fw)
            writer.writerow(processed_pdfs)

def getRawPdfs():
    raw_pdf_path = "./data/raw_pdfs/"
    return [join(raw_pdf_path, pdf) for pdf in listdir(raw_pdf_path) if isfile(join(raw_pdf_path, pdf))]

def classifyPages(path):
    reader = PdfReader(path)
    token_limit = 4096
    with open("./config/class_prompt.txt", "r") as fr:
        system_prompt = "".join(fr.readlines())

    responses = {}
    for index, page in enumerate(reader.pages):
        contents = page.extract_text()
        total_tokens = getTokens(system_prompt) + getTokens(contents)

        if index%10 == 0:
            print("Classifying page: "+str(index))
        if total_tokens < token_limit*.85:
            try:
                response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role":"system", "content":system_prompt}, {"role":"user", "content":contents}])
                if "True" in response["choices"][0]["message"]["content"] or "False" in response["choices"][0]["message"]["content"]:
                    responses[index] = response["choices"][0]["message"]["content"]
                else:
                    responses[index] = "UnexpectedResponseError"
            except:
                responses[index] = "APIErrorResponse"
        else:
            responses[index] = "TooManyTokensError"
        time.sleep(3)
    
    with open("./data/class_responses/"+path[path.rindex("/")+1:-4]+".txt", "w+") as fw:
        for page in responses:
            fw.write(str(page)+": "+responses[page]+"\n")

def trimPdf(path):
    name = path[path.rindex("/")+1:-4]
    reader = PdfReader(path)
    writer = PdfWriter()
    page_classes = []
    with open("./data/class_responses/"+name+".txt", "r") as fr:
        page_classes = list(fr.readlines())
    
    for page in page_classes:
        if "True" in page:
            writer.add_page(reader.pages[int(page[:page.index(":")])])
    writer.write("./data/trimmed_pdfs/"+name+".pdf")
    return

def PreProcessPdfs():
    with open("./config/key.txt", "r") as fr:
        openai.api_key = fr.readline()

    classified = getClassifiedPdfs()
    pdf_paths = getRawPdfs()
    start_time = time.time()
    for path in pdf_paths:
        pdf_name = path[path.rindex("/")+1:]
        if  pdf_name not in classified:
            print("Processing: "+pdf_name)
            classifyPages(path)
            trimPdf(path)
            addClassified(path)
            print("Total runtime:", time.time()-start_time)
            time.sleep(10)

if __name__ == "__main__":
    PreProcessPdfs()