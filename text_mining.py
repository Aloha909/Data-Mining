import pycld2 as cld2
import numpy as np
import spacy
from time import time
import pandas as pd

nlp_fr = spacy.load("fr_core_news_sm")
nlp_en = spacy.load("en_core_web_sm")

def words_to_list(tags: str) -> list[str]:
    tags_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
    return tags_list

def remove_common_words(tags: list[str]) -> list[str]:
    common_words: list[str] = ["lyon", "france", "symbol", "by", "uploaded", "square", "instagram", "squareformat", "iphoneography", "instagramapp", "rhônealpes", "auvergnerhônealpes", "rhone", "city", "frankreich", "francia", "ville", "nikon", "fr", "69", "photo", "villeurbanne", "rhonealpes", "photography", "canon", "live", "filter", "2013", "2014", "2010", "french", "de", "flickriosapp", "flickrmobile", "francja", "des", "lyons", "2019", "nofilter", "2012", "2016", "2015", "geotagged", "2018", "exif", "2011", "tourisme", "town", "2009", "2017", "la", "fra", "frankryk", "franța", "hdr", "eos", "creativecommons", "longexposure", "photos", "35mm", "fujifilmxt10", "tag", "22032013", "iphone", "camera"]
    tags_list = [tag for tag in tags if (not tag.isdigit()) and (tag.lower() not in common_words)]
    return tags_list

def unique_tags(tags: list[str]) -> list[str]:
    unique_tags = list(dict.fromkeys(tags))
    return unique_tags

def detect_language(text: str) -> str:
    is_reliable, _, details = cld2.detect(text, hintLanguageHTTPHeaders="fr, en", isPlainText=True, bestEffort=True)
    language = details[0][1] if is_reliable else "fr"
    return language


def lemmatize_batch(words: list[str]):
    norm = [w.strip() for w in words if w and w.strip()]
    uniq = list(dict.fromkeys(norm))

    fr_words, en_words = [], []
    for w in uniq:
        lang = detect_language(w)
        if lang == "fr":
            fr_words.append(w)
        else:
            en_words.append(w)

    out = {}

    for doc in nlp_fr.pipe(fr_words, batch_size=2000, disable=["parser", "ner", "textcat"]):
        out[doc.text] = doc[0].lemma_.lower() if len(doc) else doc.text.lower()

    for doc in nlp_en.pipe(en_words, batch_size=2000, disable=["parser", "ner", "textcat"]):
        out[doc.text] = doc[0].lemma_.lower() if len(doc) else doc.text.lower()

    return [out[w.strip()] if w and w.strip() else w for w in words]

def frequency(tags: list[str]) -> dict[str, float]:
    freq = {}
    for tag in tags:
        freq[tag] = freq.get(tag, 0) + 1
    return freq

def ner_sentence(text: str) -> list[str]:
    lang = detect_language(text)
    if lang == "fr":
        doc = nlp_fr(text)
    else:
        doc = nlp_en(text)

    entities = [ent.text for ent in doc.ents if ent.label_ in {"ORG", "GPE", "LOC", "EVENT", "FAC"}] # organisation, entite geopolotique, lieu, evenement, batiment
    return entities

def calcul_log_odds(cluster_entities: dict[int, list[str]], nb_words: int) -> dict[int, list[str]]:
    # Compte total (tous clusters)
    all_entities = []
    for entities in cluster_entities.values():
        all_entities.extend(entities)

    total_count = len(all_entities)
    global_freq = frequency(all_entities)

    cluster_top_words = {}

    for cluster, entities in cluster_entities.items():
        if not entities:
            cluster_top_words[cluster] = []
            continue

        cluster_freq = frequency(entities)
        cluster_total = len(entities)
        other_total = total_count - cluster_total

        log_odds = {}
        for entity, count in cluster_freq.items():
            p_cluster = (count + 0.5) / (cluster_total + 1)

            # Fréquence dans les autres clusters
            other_count = global_freq[entity] - count
            p_other = (other_count + 0.5) / (other_total + 1)

            log_odds[entity] = np.log(p_cluster / p_other)

        sorted_entities = sorted(log_odds.items(), key=lambda x: x[1], reverse=True)
        cluster_top_words[cluster] = [ent for ent, _ in sorted_entities[:nb_words]]

    return cluster_top_words

def top_words(df_map : pd.DataFrame, text_mining_method, nb_words: int = 3) -> dict[int, list[str]]:
    cluster_tags = {}
    for _, row in df_map.iterrows():
        cluster = row["cluster"]
        tags = row["tags"]
        if pd.isna(tags):
            continue
        tag_list = words_to_list(tags)
        if cluster not in cluster_tags:
            cluster_tags[cluster] = []
        cluster_tags[cluster].extend(tag_list)

    # cluser_tags[cluster] : liste de tous les tags du cluster

    if text_mining_method == "Lemmatisation + TF-IDF":

        all_tags = []
        for cluster in cluster_tags:
            cluster_tags[cluster] = remove_common_words(cluster_tags[cluster])
            cluster_tags[cluster] = lemmatize_batch(cluster_tags[cluster])

            all_tags.extend(unique_tags(cluster_tags[cluster]))

        all_tags_freq = frequency(all_tags)
        # le extend avec unique puis le frequency donne le nombre de clusters qui ont le tag

        cluster_top_words = {}
        for cluster in cluster_tags:
            # on fait TF-IDF
            tag_freq = frequency(cluster_tags[cluster])
            tf_idf = {}
            for tag in tag_freq:
                tf = tag_freq[tag]
                idf = np.log(len(cluster_tags) / all_tags_freq.get(tag, 1))
                tf_idf[tag] = tf * idf

            # print(tf_idf)
            # print("-----")

            # meilleur tag c'est celui avec le plus grand tf idf
            # on prend les 3 meilleurs

            sorted_tags = sorted(tf_idf.items(), key=lambda x: x[1], reverse=True)
            top_tags = [tag for tag, score in sorted_tags[:nb_words]]
            cluster_top_words[cluster] = top_tags

        return cluster_top_words

    elif text_mining_method == "NER + log-odds":
        cluster_entities = {}
        for cluster in cluster_tags:
            cluster_tags[cluster] = remove_common_words(cluster_tags[cluster])
            concat = " ".join(cluster_tags[cluster])
            entities = ner_sentence(concat)
            cluster_entities[cluster] = entities

        cluster_top_words = calcul_log_odds(cluster_entities, nb_words)

        return cluster_top_words


if __name__ == "__main__":
    print("coucou")
    t0 = time()

    tags = ["lyon, france, symbol", "by, uploaded, square, instagram", "squareformat, iphoneography, instagramapp"]
    word_tags = []

    for el in tags:
        temp_tags = words_to_list(el)
        word_tags.extend(temp_tags)


    texts: list[str] = ["lyon", "france", "symbol", "by", "uploaded", "square", "instagram", "squareformat", "iphoneography", "instagramapp", "rhônealpes", "auvergnerhônealpes", "rhone", "city", "frankreich", "francia", "ville", "nikon", "fr", "69", "photo", "villeurbanne", "rhonealpes", "photography", "canon", "live", "filter", "2013", "2014", "2010", "french", "de", "flickriosapp", "flickrmobile", "francja", "des", "lyons", "2019", "nofilter", "2012", "2016", "2015", "geotagged", "2018", "exif", "2011", "tourisme", "town", "2009", "2017", "la", "fra", "frankryk", "franța", "hdr", "eos", "creativecommons", "longexposure", "photos", "35mm", "fujifilmxt10", "tag", "22032013", "iphone", "camera"]
    texts: list[str] = ["fleur", "fleurs", "running", "ran", "courir", "couru", "maisons", "maison", "houses", "house"]
    print(lemmatize_batch(texts))
    print(lemmatize_batch(word_tags))

    # for text in texts:
    #     lem = lemmatize(text)
        # print(f"Text: {text} => Lem: {lem} ")
    print("Time taken:", time() - t0)