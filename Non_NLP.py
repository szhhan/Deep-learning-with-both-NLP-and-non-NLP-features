#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:58:13 2019

@author: sizhenhan
"""
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict


def extract_nonnlp_features(X_train3,X_test3):
    d = get_questions(X_train3,X_test3)
    
    train_nonnlp = get_hash(X_train3, d)
    test_nonnlp = get_hash(X_test3, d)
    data_nonnlp = pd.concat([train_nonnlp, test_nonnlp])
    
    kcore_dict = get_kcore_dict(data_nonnlp)
    train_nonnlp = get_kcore_features(train_nonnlp, kcore_dict)
    test_nonnlp = get_kcore_features(test_nonnlp, kcore_dict)
    
    train_nonnlp = create_minmax_feature(train_nonnlp, "kcore")
    test_nonnlp = create_minmax_feature(test_nonnlp, "kcore")
    
    neighbors = get_neighbors(data_nonnlp)
    train_nonnlp = get_neighbor_features(train_nonnlp, neighbors)
    test_nonnlp = get_neighbor_features(test_nonnlp, neighbors)
    
    frequency_map = get_frequency(data_nonnlp)
    train_nonnlp = get_freq_features(train_nonnlp, frequency_map)
    test_nonnlp = get_freq_features(test_nonnlp, frequency_map)
    train_nonnlp = create_minmax_feature(train_nonnlp, "freq")
    test_nonnlp = create_minmax_feature(test_nonnlp, "freq")
    
    cols = ["min_kcore", "max_kcore", "common_neighbor_ratio", "common_neighbor_count", "min_freq", "max_freq"]

    train_nonnlp = train_nonnlp.loc[:,cols]
    test_nonnlp = test_nonnlp.loc[:,cols]
    
    return train_nonnlp, test_nonnlp



def get_questions(train, test):
    questions_train = np.dstack([train["question1"], train["question2"]]).flatten()
    questions_test = np.dstack([test["question1"], test["question2"]]).flatten()
    questions_list = pd.DataFrame(np.append(questions_train, questions_test))[0].drop_duplicates()
    questions_list.reset_index(inplace=True, drop=True)
    return pd.Series(questions_list.index.values, index=questions_list.values).to_dict()

def get_hash(df, dic):
    df["qid1"] = df["question1"].map(dic)
    df["qid2"] = df["question2"].map(dic)
    return df.drop(["question1", "question2"], axis=1)


def get_kcore_dict(df,NB_CORES=10):
    g = nx.Graph()
    g.add_nodes_from(df.qid1)
    edges = list(zip(df["qid1"],df["qid2"]))
    g.add_edges_from(edges)
    g.remove_edges_from(g.selfloop_edges())

    df_output = pd.DataFrame(data=g.nodes(), columns=["qid"])
    df_output["kcore"] = 0
    
    for k in range(2, NB_CORES + 1):
        cores_k = nx.k_core(g, k=k).nodes()
        df_output.loc[df_output.qid.isin(cores_k), "kcore"] = k

    return df_output.to_dict()["kcore"]

def get_kcore_features(df, kcore_dict):
    df["kcore1"] = df["qid1"].map(kcore_dict)
    df["kcore2"] = df["qid2"].map(kcore_dict)
    return df


def create_minmax_feature(df, col):
    feature = np.sort(np.vstack([df[col + "1"], df[col + "2"]]).T)
    df["min_" + col] = feature[:, 0]
    df["max_" + col] = feature[:, 1]
    return df.drop([col + "1", col + "2"], axis=1)


def get_neighbors(df):
    neighbors = defaultdict(set)
    for q1, q2 in zip(df["qid1"], df["qid2"]):
        neighbors[q1].add(q2)
        neighbors[q2].add(q1)
    return neighbors


def bound(x,NEIGHBOR_UPPER_BOUND=5):
    if x["common"] > NEIGHBOR_UPPER_BOUND:
        return NEIGHBOR_UPPER_BOUND
    else:
        return x['common']
            

def get_neighbor_features(df, neighbors):
    common = df.apply(lambda x: len(neighbors[x.qid1].intersection(neighbors[x.qid2])), axis=1)
    minn = df.apply(lambda x: min(len(neighbors[x.qid1]), len(neighbors[x.qid2])), axis=1)
    df["common_neighbor_ratio"] = common / minn
    df["common"] = common
    df["common_neighbor_count"] = df.apply(lambda x: bound(x),axis=1)
    return df.drop(["common"], axis=1)


def get_frequency(df):
    all_ids = np.hstack((df["qid1"], df["qid2"]))
    counts = np.unique(all_ids,return_counts=True)
    return dict(zip(*counts))


def bound2(x,frequency_map,FREQ_UPPER_BOUND=100):
    if frequency_map[x] > FREQ_UPPER_BOUND:
        return FREQ_UPPER_BOUND
    else:
        return frequency_map[x]


def get_freq_features(df, frequency_map):
    df["freq1"] = df["qid1"].map(lambda x: bound2(x,frequency_map))
    df["freq2"] = df["qid2"].map(lambda x: bound2(x,frequency_map))
    return df
