import numpy as np
import pandas as pd
from collections import OrderedDict
from collections import Counter
import pathlib
import json
from debater_python_api.api.debater_api import DebaterApi

#Load data, remove missing values and add length column
def remove_missing_add_length(infile, outfile):
    data_df = pd.read_csv(infile)
    nonmiss_df = data_df.dropna()
    new_df = nonmiss_df.copy()
    new_df["id"] = new_df["id"].astype(str)
    new_df["id_description"] = new_df["id_description"].astype(str)
    new_df["length"] = nonmiss_df["text"].apply(lambda x: len(x))
    return new_df


#Create a list where each element is an ordered dict of a row in the dataset
def create_ordered_dicts(df):
    sentences = df.to_dict(into=OrderedDict, orient="records")
    return sentences

#Initialize argument quality, key points, and term wikifier services
def init_quality_keypoints(keyfile):
    apikey_path = pathlib.Path(keyfile)
    api_key = apikey_path.read_text().strip()
    
    debater_api = DebaterApi(apikey=api_key)
    arg_quality_cl = debater_api.get_argument_quality_client()
    keypoints_cl = debater_api.get_keypoints_client()
    term_wikifier_cl = debater_api.get_term_wikifier_client()

    return arg_quality_cl, keypoints_cl, term_wikifier_cl

#Use the argument quality service and a topic to obtain the top 1000
# sentences related to the topic
def top_1000_sentences(arg_quality_client, topic, sentences):
    #Pair every test with the topic in a dic
    sentences_topic = [
        { "sentence": sentence["text"], "topic": topic, }
        for sentence in sentences
    ]

    #Asign the scores
    scores = arg_quality_client.run(sentences_topic)

    #Sort the paired sentences by the score, from highest to lowest
    sentences_sorted = [
        s
        for s, _ in sorted(zip(sentences,scores), key=lambda x: x[1], reverse=True)
    ]

    #Keep the top 1000
    top_k = 1000
    return sentences_sorted[:top_k]


#Create structures storing the run parameters, sentences text and sentences ids
def create_sentence_structs(threshold, n_top_kps, sentences):
    run_params = {"mapping_threshold": threshold, "n_top_kps": n_top_kps}
    #List of texts...
    sentences_texts = [
        sentence["text"]
        for sentence in sentences
    ]
    #...and list with their respective ids
    sentences_ids = [
        sentence["id"]
        for sentence in sentences
    ]
    return run_params, sentences_texts, sentences_ids


#Perform the key point analysis
def key_point_analysis(keypoints_client, domain, run_params, sentences_texts, sentences_ids):
    #Clear and load the data
    keypoints_client.delete_domain_cannot_be_undone(domain)
    
    keypoints_client.upload_comments(
        domain=domain,
        comments_ids=sentences_ids,
        comments_texts=sentences_texts,
        dont_split=True,
    )
    
    keypoints_client.wait_till_all_comments_are_processed(domain=domain)

    #Run the keypoints analysis
    future = keypoints_client.start_kp_analysis_job(
        domain=domain,
        #comments_ids=sentences_ids,
        run_params=run_params,
    )

    #Get the result
    kpa_result = future.get_result(
        high_verbosity=False,
        polling_timout_secs=5,
    )

    return kpa_result, future.get_job_id()


#Convert the key points analysis result to a data frame
def kpa_to_dataframe(kpa_result):
    matchings_rows = []

    for keypoint_matching in kpa_result["keypoint_matchings"]:
        kp = keypoint_matching["keypoint"]
        
        for match in keypoint_matching["matching"]:
            match_row = [
                kp,
                match["sentence_text"],
                match["score"],
                match["comment_id"],
                match["sentence_id"],
                match["sents_in_comment"],
                match["span_start"],
                match["span_end"],
                match["num_tokens"],
                match["argument_quality"],
            ]
        
        matchings_rows.append(match_row)
        
    cols = [
        "kp",
        "sentence_text",
        "match_score",
        "comment_id",
        "sentence_id",
        "sents_in_comment",
        "span_start",
        "span_end",
        "num_tokens",
        "argument_quality",
    ]
    
    return pd.DataFrame(matchings_rows, columns=cols)


#Merge the kpa results with the sentences in the dataset
def merge_kpa(match_df, data_df, filename):
    df_merge = match_df.merge(
        data_df[["id", "id_description", "medical_specialty_new"]],
        left_on = "comment_id",
        right_on = "id",
        validate = "one_to_one",
    )

    df_merge.to_csv(filename, index=False)


#Gives a list of terms (in wikipedia) associated with each of the texts supplied
def get_sentence_to_mentions(term_wikifier_client, sentences_texts):
    mentions_list = term_wikifier_client.run(sentences_texts)
    sentence_to_mentions = {}

    for sentence_text, mentions in zip(sentences_texts, mentions_list):
        sentence_to_mentions[sentence_text] = set([
            mention["concept"]["title"]
            for mention in mentions
        ])
    
    return sentence_to_mentions

#Tally the references in wikipedia for each key point in the KPA
def get_wikipedia_terms(term_wikifier_client, merge_df):
    terms = {}
    
    for kp in set(merge_df["kp"].values):
        sentence_to_mentions = get_sentence_to_mentions(
            term_wikifier_client,
            merge_df["sentence_text"][merge_df["kp"] == kp].values
        )
        
        all_mentions = [
            mention
            for sentence in sentence_to_mentions
            for mention in sentence_to_mentions[sentence]
        ]
        
        term_count = dict(Counter(all_mentions))
        
        if "History" in term_count.keys():
            term_count.pop("History")
            
        terms[kp] = term_count

    return terms

#Dump to json file
def dump_to_json(terms, filename):
    with open(filename, "w") as fp:
        json.dump(terms, fp)
