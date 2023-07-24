import json
from rouge import Rouge
from rouge import FileRouge

def score_two_files(hyp_path, ref_path, avg_score = 0):
    """
    :param hyp_path:
    :param ref_path:
    :param avg_score:
    :return:
    """
    files_rouge = FileRouge()
    if avg_score:
        scores = files_rouge.get_scores(hyp_path,ref_path, avg=True)
    else:
        scores = files_rouge.get_scores(hyp_path, ref_path)
    return scores


score_two_files('./data/amzn_earningtrans.txt', './data2/AAPL_Q2_2023.txt', 0)