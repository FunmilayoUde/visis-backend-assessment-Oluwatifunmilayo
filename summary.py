from transformers import pipeline
import torch
from cleantext import clean
from pathlib import Path
import logging
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
das_logfile = Path.cwd() / "summarize_tokenbatches.log"

_device = 0 if torch.cuda.is_available() else -1

logging.basicConfig(
    level=logging.INFO,
    filename=das_logfile,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S",
)

path = r'C:\Users\Chimdi\Downloads\Sight'
model = AutoModelForSeq2SeqLM.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)
token_batch_length = 2048 
batch_stride = 20

session_settings = {}
session_settings['token_batch_length'] = token_batch_length
session_settings['batch_stride'] = batch_stride
number_beams = 8 
min_length =  32
max_len_ratio = 6 
length_penalty =  0.5

settings = {
    'min_length':32,
    'max_length':int(token_batch_length//max_len_ratio),
    'no_repeat_ngram_size':3, 
    'encoder_no_repeat_ngram_size' :4,
    'repetition_penalty':3.7,
    'num_beams':number_beams,
    'length_penalty':length_penalty,
    'early_stopping':True,
    'do_sample':False,
}
logging.info(f"using textgen params:\n\n:{settings}")
session_settings['num_beams'] = number_beams
session_settings['length_penalty'] = length_penalty
session_settings['max_len_ratio'] = max_len_ratio

def summarize_and_score(ids, mask, **kwargs):


    ids = ids[None, :]
    mask = mask[None, :]
    
    input_ids = ids
    attention_mask = mask
    global_attention_mask = torch.zeros_like(attention_mask)
    # put global attention on <s> token
    global_attention_mask[:, 0] = 1

    summary_pred_ids = model.generate(
            input_ids, 
            attention_mask=attention_mask, 
            global_attention_mask=global_attention_mask, 
            output_scores=True,
            return_dict_in_generate=True,
            **kwargs
        )
    summary = tokenizer.batch_decode(
                summary_pred_ids.sequences, 
                skip_special_tokens=True,
                remove_invalid_values=True,
            )
    score = round(summary_pred_ids.sequences_scores.cpu().numpy()[0], 4)
    
    return summary, score
    
def summarize_via_tokenbatches(
        input_text:str,
        batch_length=8192,
        batch_stride=16,
        **kwargs,
    ):
    
    encoded_input = tokenizer(
                        input_text, 
                        padding='max_length', 
                        truncation=True,
                        max_length=batch_length, 
                        stride=batch_stride,
                        return_overflowing_tokens=True,
                        add_special_tokens =False,
                        return_tensors='pt',
                    )
    
    in_id_arr, att_arr = encoded_input.input_ids, encoded_input.attention_mask
    gen_summaries = []

    for _id, _mask in zip(in_id_arr, att_arr):

        result, score = summarize_and_score(
            ids=_id, 
            mask=_mask, 
            **kwargs,
        )
        score = round(float(score),4)
        _sum = {
            "input_tokens":_id,
            "summary":result,
            "summary_score":score,
        }
        gen_summaries.append(_sum)
        sum_text = [s["summary"][0] for s in gen_summaries]
        sum_scores = [f"\n - {round(s['summary_score'],4)}" for s in gen_summaries]
        scores_text = "\n".join(sum_scores)
        full_summary = "\n\t".join(sum_text)
        

    

    return full_summary

