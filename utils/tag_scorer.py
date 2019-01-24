import argparse
import numpy as np
from tqdm import tqdm
from utils.tagging import AspectTagger, InterestTagger


def score(ref_lines, gen_lines, tagger):
    corrects = []
    for ref, gen in tqdm(zip(ref_lines, gen_lines)):
        corrects.append(tagger.tag(ref) == tagger.tag(gen))

    return np.mean(corrects)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ref', type=str)
    parser.add_argument('gen', type=str)
    parser.add_argument('tagger', type=str)
    args = parser.parse_args()
    with open(args.ref) as f:
        ref_lines = f.read().strip().split('\n')
    with open(args.gen) as f:
        gen_lines = f.read().strip().split('\n')

    assert len(ref_lines) == len(gen_lines)

    tagger = {
        'aspect_tagger': AspectTagger,
        'interest_tagger': InterestTagger,
    }[args.tagger]()

    print(f'Tagging Accuracy={score(ref_lines, gen_lines, tagger): .3f} of Generated Sentences.')
