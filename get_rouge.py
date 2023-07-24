import json
from rouge import Rouge
from rouge import FilesRouge

def score_two_files(hyp_path, ref_path, avg_score = 0):
    """
    :param hyp_path:
    :param ref_path:
    :param avg_score:
    :return:
    """
    files_rouge = FilesRouge()
    if avg_score:
        scores = files_rouge.get_scores(hyp_path, ref_path, avg=True)
    else:
        scores = files_rouge.get_scores(hyp_path, ref_path)
    return scores

file1 = './output1.txt'
file2 = './output.txt'
score_two_files(file1, file2, 0)
