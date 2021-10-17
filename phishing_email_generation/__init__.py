import random
from transformers import AutoModel, AutoTokenizer
from data_model import EnFMGItem

SPECIAL_TOKENS  = {"bos_token": "<|BOS|>",
                   "eos_token": "<|EOS|>",
                   "unk_token": "<|UNK|>",
                   "pad_token": "<|PAD|>",
                   "sep_token": "<|SEP|>"}

def _join_keywords(keywords, randomize=False):
    N = len(keywords)

    #random sampling and shuffle
    if randomize: 
        M = random.choice(range(N+1))
        keywords = keywords[:M]
        random.shuffle(keywords)

    return ','.join(keywords)

def generate(model: AutoModel, tokenizer: AutoTokenizer, item: EnFMGItem):
    kw = _join_keywords(item.keywords)
    title = "Greeting" #文體起始句
    #types = "Fraud" #犯罪的文章，包含詐騙郵件詞表
    types = "Normal" 
    category = "TECH" #商務文章，包含正常郵件詞表
    # category => 內文主題 'CRIME' ,'ENTERTAINMENT', 'POLITICS', 'SPORTS', 'BUSINESS', 'TECH', 'EDUCATION', 'HEALTHY LIVING', 'MONEY', 'CULTURE & ARTS'
    formats = "Email" #Email文體

    prompt = SPECIAL_TOKENS['bos_token'] + types + \
                SPECIAL_TOKENS['sep_token'] + category + \
                SPECIAL_TOKENS['sep_token'] + formats + \
                SPECIAL_TOKENS['sep_token'] + title + \
                SPECIAL_TOKENS['sep_token'] + kw

    generated = tokenizer.encode(prompt, return_tensors="pt")

    sample_outputs = model.generate(
        generated, 
        do_sample=True,   
        min_length=50, 
        max_length=768,
        top_k=30,                                 
        top_p=0.7,        
        temperature=0.9,
        repetition_penalty=2.0,
        num_return_sequences=1
        )
    
    sentence = []
    for i, sample_output in enumerate(sample_outputs):
        text = tokenizer.decode(sample_output, skip_special_tokens=True)
        a = len(types)+len(category)+len(formats)+ len(title) + len(','.join(keywords)) 
        sentence.append(title+text[a:])
        # print('{} {}\n\n'.format(title,sentence[i]))
    
    return sentence