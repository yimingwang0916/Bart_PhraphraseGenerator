os.getcwd()  
os.chdir('/content/gdrive/My Drive/sentence_compression/data') # files' path

# quantity process 
#path = '/content/gdrive/My Drive/sentence_compression/data'
#os.listdir(path)

#datalist = []
#for n in os.listdir(path):
#    if os.path.splitext(n)[1] == '.json':   
#        datalist.append(n)

src_sent_str = []
src_head_str = []

data_path = 'sent-comp.train01.json' # file's name

for i in range(1):
  with open(data_path, 'r', encoding="utf-8") as load:
    data = load.readlines()
    for line in data: 
      context = line
      x = context.startswith("    \"sentence\":")  # to avoid those lines including “sentence” in the middle 
      y = context.startswith("  \"headline\":")    # to avoid those lines including “headline” in the middle 
      if x == True:
        context = context.replace('\n', "").replace('\\n', "").replace('\\\n', "").replace('\\\\n', "").replace('\\\\\n', "") # delete meaningless string
        con_list = context.split()                 # delete words, "sentence" and "headline" at the beginning of strings
        con_list.remove("\"sentence\":")
        if len(src_sent_str) < 1:                  # append only non-repeat sentences
          src_sent_str.append(' '.join(con_list))  
        else:
          con_check = ' '.join(con_list)
          if con_check != src_sent_str[-1]:
             src_sent_str.append(con_check)               
      if y == True:
        context = context.replace('\n', "").replace('\\n', "").replace('\\\n', "").replace('\\\\n', "").replace('\\\\\n', "")
        con_list = context.split()
        con_list.remove("\"headline\":")
        src_head_str.append(' '.join(con_list))
  print('-- finish extraction of ' + data_path)

# save extracted text as new .json file
df_sent = pd.DataFrame(src_sent_str,columns = ['sentence']) # save as .json
df_head = pd.DataFrame(src_head_str,columns = ['headline'])
df = pd.concat([df_sent, df_head],axis=1)
print(df)
data_save_path = "/content/gdrive/My Drive/sentence_compression/preprocessed_data/train01.json" 
df.to_json(data_save_path, orient="index") 
print('-- saved processed file as' + data_save_path)
