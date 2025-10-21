##is_gpt
## file folder
##threshold = 0.4, 0.5
##compute consistency on annotation of full pragmatic inference part 

import json
import numpy as np
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from scipy.optimize import linear_sum_assignment

# load sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def parse_annotations(cell, is_gpt=False):
    print("type of input:", type(cell))
    try:
        if isinstance(cell, str):
            if cell.strip().lower() in ["literal", "none"]:  #necessary, literal in Pragmatic_Inferences
                return "literal"
            if not is_gpt: #if human
            # ensure human annotations are wrapped in a list
                cell = "[" + cell.strip().strip(",") + "]" #human annotated:{}, {}->[{}, {}]
        ann = json.loads(cell)
        print("json loaded ann:", ann)
        if isinstance(ann, dict) and is_gpt: #json format, gpt annotation output
            print("ann is a dict!")
            # GPT annotations: dict with numbered keys
            return [
                {"content": v["content"], "type": v["type"], "confidence": v.get("confidence", 1.0)}
                for v in ann.values()
            ]
        elif isinstance(ann, list):
            print("ann is a list!")
            # Human annotations: list of dicts; if already list, do nothing
            return ann
    except Exception as e:
        print("Parse error:", e, "Cell:", cell)
        return []
    return []



def compare_inferences(human_list, gpt_list, threshold=0.50, type_error_penalty=0.8, type_soft=True):
    # Handle "literal" cases
    if human_list == "literal" and gpt_list == "literal":
        return {"precision":1,"recall":1,"f1":1,"semantic":1,"type_acc":1}
    if human_list == "literal" or gpt_list == "literal": #pragmatic inference presence
        return {"precision":0,"recall":0,"f1":0,"semantic":0,"type_acc":0}
    
    H, G = len(human_list), len(gpt_list)
    if H == 0 or G == 0:
        return {"precision":0,"recall":0,"f1":0,"semantic":0,"type_acc":0}
    
    # embeddings
    human_emb = model.encode([h["content"] for h in human_list], normalize_embeddings=True)
    gpt_emb = model.encode([g["content"] for g in gpt_list], normalize_embeddings=True)

    # similarity matrix
    ## Computes a similarity matrix between every human inference embedding and every GPT inference embedding.
    ## similarity between human inference i and GPT inference j.
    sim_matrix = cosine_similarity(human_emb, gpt_emb)
    print("cosine similarity matrix:", sim_matrix)
    # adjust for type match (soft or hard)
    if type_error_penalty != 1.0:
        for i, h in enumerate(human_list):
            for j, g in enumerate(gpt_list):
                if h["type"] != g["type"]:
                    sim_matrix[i, j] *= type_error_penalty if type_soft else 0


    ####matches = [(i, j, sim_matrix[i,j]) for i, j in zip(row_ind, col_ind)]
    human_matched = sum(sim_matrix[i,:].max() >= threshold for i in range(sim_matrix.shape[0])) ##cosine Sim over a threshold in human annotation
    recall = human_matched / sim_matrix.shape[0] if sim_matrix.shape[0] > 0 else 0 #sum of max cos_sim averaged over all the messages
    
    gpt_matched = sum(sim_matrix[:,j].max() >= threshold for j in range(sim_matrix.shape[1])) ##cosine Sim over a threshold for another Annotator
    precision = gpt_matched / sim_matrix.shape[1] if sim_matrix.shape[1] > 0 else 0
    # filter by threshold
    ####matches = [m for m in matches if m[2] >= threshold]
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    
    # highest semantic similarity summed over all the messages (average best match scores, not just matches)
    semantic = (
    (sum(max(sim_matrix[i, :]) for i in range(H)) +
     sum(max(sim_matrix[:, j]) for j in range(G)))
    / (H + G)
    if (H + G) > 0 else 0)

    # type accuracy: among GPTs that matched, how many have the right type
    type_acc_count = 0
    type_acc_total = 0
    for i in range(H):
        j_best = np.argmax(sim_matrix[i, :])  # best GPT match for human annotation for i
        if sim_matrix[i, j_best] >= threshold:
            type_acc_total += 1  #matched inferences 
            if human_list[i]["type"] == gpt_list[j_best]["type"]: #matched inferences with type match
                type_acc_count += 1
    type_acc = type_acc_count / type_acc_total if type_acc_total > 0 else 0

    return {"precision":precision,"recall":recall,"f1":f1,
            "semantic":semantic,"type_acc":type_acc}

folder_A = " "
folder_B = "  "
# Collect basenames
# os.path.splitext(f) returns a tuple: (root, extension).
files_A = {os.path.splitext(f)[0].split("_")[0] for f in os.listdir(folder_A) if f.endswith(".csv")}
print("gpt filenames:", files_A)
###files_B = {os.path.splitext(f)[0] for f in os.listdir(folder_B) if f.endswith(".csv")}
files_B = {os.path.splitext(f)[0].split("_")[0] for f in os.listdir(folder_B) if f.endswith(".csv")}

common_files = files_A & files_B
print("common files:", common_files)
print(len(common_files))

corpus_results = []
#macro-avg
for name in common_files:
    print("name", name)
    path_A = os.path.join(folder_A, f"{name}.csv")  
    #path_A = os.path.join(folder_A, f"{name}.csv")  
    path_B = os.path.join(folder_B, f"{name}.csv")
    #path_B = os.path.join(folder_B, f"{name}_merged.csv")
    df_A = pd.read_csv(path_A)
    df_B = pd.read_csv(path_B)
    merged = pd.merge(df_A, df_B, on="Message_ID", suffixes=("_A", "_B"))
    # --- handle most_salient_inference with SBERT ---
    # most salient in gpt A: indexed; human B directly extracted
    # not possible to have empty entries
    thread_result = []
    for row_id, row in merged.iterrows(): #both A and B, if 
        #salient_key_A = row["most_salient_inference_A"]
        machine_full_inferences = row["Pragmatic_Inferences_A"]
        human_full_inferences = row["Pragmatic_Inferences_B"]
        if pd.isna(human_full_inferences) or pd.isna(machine_full_inferences): # if has no annotations in full inferences-->moderators' message, skip the whole line
            continue
        machine_inferences = parse_annotations(machine_full_inferences, is_gpt=True)
        human_inferences = parse_annotations(human_full_inferences, is_gpt=False) ##should change
        msg_result = compare_inferences(human_inferences, machine_inferences, threshold=0.5, type_error_penalty=1.0, type_soft=True)
        #{"precision":0,"recall":0,"f1":0,"semantic":0,"type_acc":0}
        thread_result.append(msg_result)

    macro_thread = {
    'precision': np.mean([m['precision'] for m in thread_result]),
    'recall': np.mean([m['recall'] for m in thread_result]),
    'f1': np.mean([m['f1'] for m in thread_result]),
    'semantic': np.mean([m['semantic'] for m in thread_result]),
    'type_acc': np.mean([m['type_acc'] for m in thread_result]),
    }
    corpus_results.append({"file": name, **macro_thread})

results_df = pd.DataFrame(corpus_results)
print(results_df)
col_means = results_df.mean(axis=0, numeric_only=True)
print("mean IAA:", col_means)
results_df.loc['mean'] = results_df.mean(axis=0, numeric_only=True)
#results_df.to_csv("IAA_humanSelfConsistency0.4_allInference_mean.csv", index=False)

  
''' 
# Example usage
human_cell = '{"content":"The OP likes Taylor ...","type":"expressives"}, {"content":"The OP\'s post might...","type":"representatives"}'
gpt_cell = '{"1":{"content":"The responder believes ...","type":"representatives","confidence":0.93}, "2":{"content":"The responder is dismissive...","type":"expressives","confidence":0.90}}'

human_anns = parse_annotations(human_cell, is_gpt=False)
gpt_anns = parse_annotations(gpt_cell, is_gpt=True)
print("human annotation parsed:", human_anns)
print("gpt annotations parsed:", gpt_anns)

print(compare_inferences(human_anns, gpt_anns, type_soft=True))
'''
