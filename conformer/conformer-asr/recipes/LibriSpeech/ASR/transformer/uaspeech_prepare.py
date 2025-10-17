"""
Data preparation.

Download: http://www.openslr.org/12

Author
------
Mirco Ravanelli, Ju-Chieh Chou, Loren Lugosch 2020
"""

import os
import csv
import random
from collections import Counter
from dataclasses import dataclass
import functools
import logging
from speechbrain.utils.data_utils import download_file, get_all_files
from speechbrain.dataio.dataio import (
    load_pkl,
    save_pkl,
    merge_csvs,
    read_audio_info,
)
from speechbrain.utils.parallel import parallel_map

import pandas as pd

logger = logging.getLogger(__name__)
OPT_FILE = "opt_uaspeech_prepare.pkl"
SAMPLERATE = 16000


def prepare_uaspeech(
    data_folder,
    save_folder,
    tr_splits=[],
    dev_splits=[],
    te_splits=[],
    select_n_sentences=None,
    merge_lst=[],
    merge_name=None,
    create_lexicon=False,
    skip_prep=False,
    block_filter=None,  # <— filter for blocks and speakers
):
    if skip_prep:
        return

    splits = tr_splits + dev_splits + te_splits
    conf = {
        "select_n_sentences": select_n_sentences,
        "block_filter": block_filter,
    }

    # Saving folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Check if this phase is already done (if so, skip it)
    save_opt = os.path.join(save_folder, OPT_FILE)
    if skip(splits, save_folder, conf):
        logger.info("Skipping preparation, completed in previous run.")
        return
    else:
        logger.info("Data_preparation...")

    # read wordlist to dict
    text_dict = {}
    word_xls_path = os.path.join(data_folder, "speaker_wordlist.xls")
    word_xls = pd.read_excel(word_xls_path, sheet_name="Word_filename", header=0)
    for i in range(word_xls.shape[0]):
        word, word_key = word_xls.iloc[i].values
        text_dict[word_key] = word.upper()

    print(text_dict)

    # create csv files for each split
    for split_index, split in enumerate(splits):
        print(split)

        split_by_block = split.startswith('B')
        if split_by_block:
            wav_lst = get_all_files(
                os.path.join(data_folder, "audio"), match_and=[".wav", f"_{split}_"], exclude_or=["control"]
            )

            # plug for control blocks
            control_lst = get_all_files(
                os.path.join(data_folder, "audio", "control"), match_and=[".wav", f"_{split}_"]
            )
            create_csv(
                save_folder, control_lst, text_dict, split + "-control", len(control_lst),
            )
        else:  # split by speakers
            if split.startswith('C'):
                wav_lst = get_all_files(
                    os.path.join(data_folder, "audio"), match_and=[".wav", f"{split}_"]
                )
            else:
                wav_lst = get_all_files(
                    os.path.join(data_folder, "audio"), match_and=[".wav", f"{split}_"], exclude_or=["control"]
                )
        if block_filter is not None:
                tag = f"_{block_filter}_"
                wav_lst = [p for p in wav_lst if tag in os.path.basename(p)]

            
        if select_n_sentences is not None:
            n_sentences = select_n_sentences[split_index]
        else:
            n_sentences = len(wav_lst)
        print(wav_lst[0])

        create_csv(
            save_folder, wav_lst, text_dict, split, n_sentences,
        )

    # Merging csv file if needed
    if merge_lst and merge_name is not None:
        for category in ["", "-control"] if split_by_block else [""]:
            merge_files = [split_ua + category + ".csv" for split_ua in merge_lst]
            dest_file = merge_name.replace(".", category+".") 
            merge_csvs(
                data_folder=save_folder, csv_lst=merge_files, merged_csv=dest_file
            )

    # saving options
    save_pkl(conf, save_opt)


@dataclass
class LSRow:
    snt_id: str
    spk_id: str
    duration: float
    file_path: str
    words: str


def process_line(wav_file_path, text_dict) -> LSRow:
    wav_file = os.path.basename(wav_file_path)
    wav_filename = os.path.splitext(wav_file)[0]

    snt_id = wav_filename
    spk_id, block, word_key, mic = wav_filename.split('_')

    if word_key.startswith('U'):
        word_key = '_'.join([block, word_key])

    if word_key not in text_dict:
        print(f"{word_key} not in text_dict")
        print(wav_file_path)
    wrds = text_dict[word_key]

    print(f"read_audio_info({wav_file_path})")
    try:
        info = read_audio_info(wav_file_path)
        duration = info.num_frames / info.sample_rate
    except RuntimeError:
        return None
    print(f"duration: {duration})")

    return LSRow(
        snt_id=snt_id,
        spk_id=spk_id,
        duration=duration,
        file_path=wav_file_path,
        words=wrds,
    )


def create_csv(
    save_folder, wav_lst, text_dict, split, select_n_sentences,
):
    """
    Create the dataset csv file given a list of wav files.

    Arguments
    ---------
    save_folder : str
        Location of the folder for storing the csv.
    wav_lst : list
        The list of wav files of a given data split.
    text_dict : list
        The dictionary containing the text of each sentence.
    split : str
        The name of the current data split.
    select_n_sentences : int, optional
        The number of sentences to select.

    Returns
    -------
    None
    """
    # Setting path for the csv file
    csv_file = os.path.join(save_folder, split + ".csv")
    if os.path.exists(csv_file):
        logger.info("Csv file %s already exists, not recreating." % csv_file)
        return
    msg = "Creating csv lists in  %s..." % (csv_file)
    logger.info(msg)

    csv_lines = [["ID", "duration", "wav", "spk_id", "wrd"]]

    snt_cnt = 0
    line_processor = functools.partial(process_line, text_dict=text_dict)
    print(f"processing {len(wav_lst)}")
    # Processing all the wav files in wav_lst
    # FLAC metadata reading is already fast, so we set a high chunk size
    # to limit main thread CPU bottlenecks
    for row in parallel_map(line_processor, wav_lst, chunk_size=8192):
        if not row:
            continue

        csv_line = [
            row.snt_id,
            str(row.duration),
            row.file_path,
            row.spk_id,
            row.words,
        ]

        # Appending current file to the csv_lines list
        csv_lines.append(csv_line)

        snt_cnt = snt_cnt + 1

        # parallel_map guarantees element ordering so we're OK
        if snt_cnt == select_n_sentences:
            break

    # Writing the csv_lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    # Final print
    msg = "%s successfully created!" % (csv_file)
    logger.info(msg)


def skip(splits, save_folder, conf):
    """
    Detect when the uaspeech data prep can be skipped.

    Arguments
    ---------
    splits : list
        A list of the splits expected in the preparation.
    save_folder : str
        The location of the seave directory
    conf : dict
        The configuration options to ensure they haven't changed.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # Checking csv files
    skip = True

    for split in splits:
        if not os.path.isfile(os.path.join(save_folder, split + ".csv")):
            skip = False

    #  Checking saved options
    save_opt = os.path.join(save_folder, OPT_FILE)
    if skip is True:
        if os.path.isfile(save_opt):
            opts_old = load_pkl(save_opt)
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False

    return skip

if __name__ == "__main__":
    CONTROL_TRAIN_LIST = ["F02", "F03", "F04", "F05",
            "M01", "M04", "M05", "M07", "M08", "M09", "M10","M12", "M14", "M16"]
    prepare_uaspeech(
        "/mnt/DataSets/UASpeech/",
        "/home/zsim710/partitions/uaspeech/by_speakers/",
        CONTROL_TRAIN_LIST,
        ["M11"],
        ["M11"],
        merge_lst=CONTROL_TRAIN_LIST,
        merge_name="train.csv",
        block_filter="B3",
    )
    
#    prepare_uaspeech(
#        "/data/qwan121/UASpeech",
#        "/home/qwan121/partitions/uaspeech/by_blocks",
#        ["B1", "B2"],
#        ["B3"],
#        ["B3"],
#        merge_lst=["B1", "B2"],
#        merge_name="train.csv"
#    )
