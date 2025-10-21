
##compute statistical correlation between variables and "aggressiveness"

from compute_corpus_stats import parse_annotations
from collections import defaultdict
#from scipy.stats.contingency import association
from scipy.stats import pointbiserialr
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif

import numpy as np
import os
import pandas as pd


def extract_speech_act_features(inf_list):
    counts = {"representatives": 0, "directives": 0, "commissives": 0,
              "expressives": 0, "declarations": 0}
    if not inf_list:
        return None
    if inf_list in ['literal']: #even if literal, still has "aggressive" label 
        return counts
    for inf in inf_list:  #a list of dicts, with "type" as a key
        t = inf.get("type", None) 
        if t in counts:
            counts[t] += 1
    return counts


def computeStats(folder_root, if_gpt=False):
    total_inference_type = defaultdict(list)
    for file in os.listdir(folder_root):
        if file.endswith(".csv"):
            filename = os.path.splitext(os.path.basename(file))[0]
            print("file name:", filename) 
            msg_level_type_count_dict = []
            file_path = os.path.join(folder_root, file)
            df = pd.read_csv(file_path)
            print("df length:", len(df))
            for idx, row in df.iterrows():
                val = row["Pragmatic_Inferences"]
                if pd.isna(val):
                    continue  # skip empty cells
                prag_inferences = parse_annotations(str(val), is_gpt=if_gpt)
                #print("prag inferences:", prag_inferences)
                type_counts_per_msg = extract_speech_act_features(prag_inferences) #one message (all pragmatic inferences) has one dict 
                msg_level_type_count_dict.append(type_counts_per_msg)
            ##file_inference_type.append(msg_level_type_count_dict)
            print("len of prag counts:", len(msg_level_type_count_dict))
            total_inference_type[filename] = msg_level_type_count_dict
    return total_inference_type #a dictionary of a list of dictionaries 



def assign_turn_and_depth(msg_id, turn=0): #turn: from root going down until to the leaf for each subthread; subthread_depth:distance from leaf 
    children = df[df['Reply_To'] == msg_id]['Message_ID'].tolist()
    if not children:
        # leaf node
        df.loc[df['Message_ID'] == msg_id, 'turn_index'] = turn
        df.loc[df['Message_ID'] == msg_id, 'subthread_depth'] = 0  
        return 0
    else:
        df.loc[df['Message_ID'] == msg_id, 'turn_index'] = turn
        max_depth = 0
        for child_id in children:
            child_depth = assign_turn_and_depth(child_id, turn + 1)
            max_depth = max(max_depth, child_depth)
        df.loc[df['Message_ID'] == msg_id, 'subthread_depth'] = max_depth + 1
        return max_depth + 1

human_iaa_path = "  "
gpt4_corpus_path = "  "
gpt5_corpus_path = "  "

mi_list = []
for file in os.listdir(human_iaa_path):
    if file.endswith(".csv"):
        filename = os.path.splitext(os.path.basename(file))[0]
        print("filename:", filename)
        prag_inference_type_count = total_inference_type_count[filename] #each file as a list of dictionaries 
        file_path = os.path.join(human_iaa_path, file)
        
        df = pd.read_csv(file_path)
        df = df.dropna(subset=["aggressive"]) #if no annotation on aggressive, skip the line (no prag inference as well, not even literal)-->moderator's

        df['prag_type_count'] = prag_inference_type_count
        df_prag_type_count = df["prag_type_count"].apply(pd.Series)
        df = pd.concat([df, df_prag_type_count], axis=1)
        #print(df['Subreddit'])
        df['turn_index'] = 0
        df['subthread_depth'] = 0
        
        msg_dict = df.set_index('Message_ID').to_dict('index')
        root_messages = df[df['Reply_To'].isna()]['Message_ID'].tolist() #root message has no "reply_to"

        for root_id in root_messages:
            df.loc[df['Message_ID'] == root_id, 'turn_index'] = 0
            assign_turn_and_depth(root_id, 0)
        df['parent_username'] = df['Reply_To'].map(lambda x: msg_dict[x]['Message_Author'] if pd.notna(x) else None)
        df['aggressive_binary'] = df['aggressive'].map({'NAG':0, 'CAG':1, 'OAG':1})  #'CAG':'AGG', 'OAG': 'AGG'}) 
        ##full_prag_infer = pd.DataFrame(prag_inference_type_count)
        #print("turn index:", df['turn_index'])
        #print("subthread depth:", df['subthread_depth'])
        numeric_features = ["representatives", "directives", "commissives", "expressives", "declarations", 'turn_index', 'subthread_depth']
        categorical_features = ["inference_type", "as_intended", "PRE/IMP", 'parent_username', 'Message_Author']
        all_features = numeric_features + categorical_features

        df_encoded = df[numeric_features].copy()
        for col in categorical_features:
            df_encoded.loc[:, col] = LabelEncoder().fit_transform(df[col].astype(str))
        
        ##df = df.dropna(subset=["aggressive_binary"])
        ##df["aggressive_binary_shuffled"] = np.random.permutation(df["aggressive_binary"].values)
        y = df['aggressive_binary']  ##df["aggressive_binary_shuffled"] 
        print("y:", y)
        # Compute mutual information
        mi_scores = mutual_info_classif(df_encoded, y, 
                                  discrete_features=[False]*len(numeric_features) + [True]*len(categorical_features), 
                                  random_state=42)
        mi_list.append(mi_scores)
        
        
# Create results DataFrame
# Convert to DataFrame
mi_array = np.array(mi_list)
mi_mean = mi_array.mean(axis=0)
mi_std = mi_array.std(axis=0)

mi_results = pd.DataFrame({
    "Feature": all_features,
    "MI_Mean": mi_mean,
    "MI_STD": mi_std
}).sort_values(by="MI_Mean", ascending=False)
print("Mutual information on all the variables:")
print(mi_results)


