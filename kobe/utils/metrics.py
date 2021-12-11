import codecs
import os
import subprocess
import sys


def bleu(reference, candidate, log_path, print_log, config, lang="de", bpe=False):
    """
    return the bleu score, including multi-bleu and nist bleu.
    :param reference: reference
    :param candidate: candidate
    :param log_path: path to log
    :param print_log: function to print log
    :param config: configuration
    :param lang: language
    :param bpe: use byte-pair encoding or not
    :return: bleu score
    """
    # check if there is specified reference file
    # if not, create one
    if config.refF != "":
        ref_file = config.refF
    else:
        ref_file = log_path + "reference.txt"
        with codecs.open(ref_file, "w", "utf-8") as f:
            for s in reference:
                if not config.char:
                    f.write(" ".join(s) + "\n")
                else:
                    f.write("".join(s) + "\n")
    cand_file = log_path + "candidate.txt"
    with codecs.open(cand_file, "w", "utf-8") as f:
        for s in candidate:
            if not config.char:
                f.write(" ".join(s).strip() + "\n")
            else:
                f.write("".join(s).strip() + "\n")
    # file to store results
    temp = log_path + "result.txt"
    # implementation for German, nist bleu
    # if bpe:
    #     cand_wobpe = log_path+"cand_wobpe.txt"
    #     cmd_bpe_remove = "cat " + cand_file + \
    #         " | sed -E 's/(@@ )|(@@ ?$)//g' > " + cand_wobpe
    #     os.system(cmd_bpe_remove)
    # detok_ref_file = log_path+'detok_reference.txt'
    # detok_cand_file = log_path+'detok_candidate.txt'
    # cmd_detok_ref = "perl script/detokenize.perl -l " + \
    #     lang + " < " + ref_file + " > " + detok_ref_file
    # cmd_detok_cand = "perl script/detokenize.perl -l " + \
    #     lang + " < " + cand_wobpe + " > " + detok_cand_file
    # os.system(cmd_detok_ref)
    # os.system(cmd_detok_cand)
    # command = "perl script/multi-bleu-detok.perl " + \
    #     detok_ref_file + "<" + detok_cand_file + "> " + temp
    # run the multi-bleu perl script and get the score
    command = (
        "perl core/utils/multi-bleu.perl -lc "
        + ref_file
        + "<"
        + cand_file
        + "> "
        + temp
    )

    try:
        subprocess.call(command, shell=True)
    except OSError as e:
        print("Execution failed:", e, file=sys.stderr)

    with open(temp) as ft:
        result = ft.read()
    os.remove(temp)
    # print the score
    print_log(result)

    return float(result.split()[2][:-1])
