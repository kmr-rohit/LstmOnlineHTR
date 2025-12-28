"""Evaluation metrics for HTR."""

import editdistance
import numpy as np


def calculate_cer(predictions, targets):
    """
    Calculate Character Error Rate.

    Args:
        predictions: List of predicted strings
        targets: List of target strings
    Returns:
        CER as percentage
    """
    total_chars = 0
    total_errors = 0

    for pred, target in zip(predictions, targets):
        errors = editdistance.eval(pred, target)
        total_errors += errors
        total_chars += len(target)

    cer = (total_errors / total_chars) * 100 if total_chars > 0 else 0
    return cer


def calculate_wer(predictions, targets):
    """
    Calculate Word Error Rate.

    Args:
        predictions: List of predicted strings
        targets: List of target strings
    Returns:
        WER as percentage
    """
    total_words = 0
    total_errors = 0

    for pred, target in zip(predictions, targets):
        pred_words = pred.split()
        target_words = target.split()

        errors = editdistance.eval(pred_words, target_words)
        total_errors += errors
        total_words += len(target_words)

    wer = (total_errors / total_words) * 100 if total_words > 0 else 0
    return wer


def ctc_greedy_decode(output, blank=0):
    """
    CTC greedy decoding - collapse repeated characters and remove blanks.

    Args:
        output: Sequence of class indices (seq_len,)
        blank: Index of CTC blank token
    Returns:
        Decoded sequence without blanks and repetitions
    """
    result = []
    prev = -1

    for idx in output:
        if idx != blank and idx != prev:
            result.append(idx)
        prev = idx

    return result
