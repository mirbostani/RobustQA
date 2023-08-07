from tqdm import tqdm
import copy
import json
import argparse

SQUAD_TRAIN = "train-v1.1.json"
SQUAD_TRAIN_ADV_ONLY = "adv.json"
SQUAD_TRAIN_ADV_MERGED = "train-v1.1-adv.json"

def parse_args():
    parser = argparse.ArgumentParser("SQuAD Adversarial Merger", add_help=True)
    parser.add_argument(
        "--train",
        type=str,
        default=SQUAD_TRAIN,
        help="SQuAD v1.1 training dataset file."
    )
    parser.add_argument(
        "--adversary",
        type=str,
        default=SQUAD_TRAIN_ADV_ONLY,
        help="Generated adversarial examples from SQuAD v1.1 dataset."
    )
    parser.add_argument(
        "--merged",
        type=str,
        default=SQUAD_TRAIN_ADV_MERGED,
        help="The merged SQuAD v1.1 training dataset with generated adversarial examples."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=8759,
        help="Limit number of used adversarial examples."
    )
    args = parser.parse_args()
    return args

def gen_adv(adv_data, qid):
    for adv in adv_data:
        if qid == adv["orig_id"]:
            adv_example = {
                "answers": adv["answers"],
                "question": adv["question"],
                "id": adv["id"]
            }
            # print(qid, adv["orig_id"], adv["id"])
            return adv_example

    return None

def main():
    args = parse_args()
    SQUAD_TRAIN = args.train
    SQUAD_TRAIN_ADV_ONLY = args.adversary
    SQUAD_TRAIN_ADV_MERGED = args.merged

    af = open(SQUAD_TRAIN_ADV_ONLY, "r")
    adv_data = json.load(af)
    adv_kv = {}
    for adv in adv_data:
        adv_kv[adv["orig_id"]] = adv

    f = open(SQUAD_TRAIN, "r")
    data = json.load(f)
    cp_data = copy.deepcopy(data)

    limit = args.limit # 8759 out of 87599 total
    total = 0
    count = 0

    for iarticle, article in tqdm(enumerate(data["data"]), total=len(data["data"])):
        for iparagraph, paragraph in enumerate(article["paragraphs"]):
            for iqa, qa in enumerate(paragraph["qas"]):
                qid = qa["id"]
                total += 1 # if 'break' is activated, only one instance of a paragraph is used, others are skipped

                if limit != None and count < limit:
                    adv_example = adv_kv.get(qid)
                    if adv_example:
                        adv_ex = {
                            "answers": [{"text": text, "answer_start": answer_start} for text, answer_start in list(zip(adv_example["answers"]["text"], adv_example["answers"]["answer_start"]))],
                            "question": adv_example["question"],
                            "id": adv_example["id"]
                        }
                        cp_data["data"][iarticle]["paragraphs"][iparagraph]["qas"].append(adv_ex)
                        count += 1
                        # break # add adv for each paragraph (max 18403 paragraph)

    print(f"total: {total}")
    print(f"adv added: {count}")      

    with open(SQUAD_TRAIN_ADV_MERGED, "w") as mf:
        json.dump(cp_data, mf)                  

    f.close()
    af.close()


if __name__ == "__main__":
    main()
