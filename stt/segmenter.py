"""Découpage de l'audio en segments de parole (machine à états pure).

Ce module ne dépend ni du micro, ni de torch, ni de Whisper : il reçoit des
blocs audio et une fonction qui donne la probabilité de parole d'un bloc.
Il peut donc être testé sans matériel (voir tests/test_segmenter.py) et
rejouer un fichier audio avec ``segment_audio``.
"""

import collections

import numpy as np

from .config import (
    BLOCK_SECONDS,
    BLOCK_SIZE,
    DEFAULT_SILENCE_MS,
    DEFAULT_VAD_THRESHOLD,
    MAX_SEGMENT_SECONDS,
    MIN_SPEECH_SECONDS,
    NEGATIVE_MARGIN,
    PRE_ROLL_SECONDS,
    START_BLOCKS,
    TAIL_SECONDS,
)


class Segmenter:
    """Transforme un flux de blocs audio en segments de parole.

    Par rapport à la version d'origine :
    - la durée minimale ne compte que les blocs réellement voisés (le pré-roll
      et le silence final n'en font plus partie) ;
    - le silence final est retiré, hormis une courte queue (TAIL_SECONDS).
    """

    def __init__(
        self,
        probability,
        reset=None,
        threshold=DEFAULT_VAD_THRESHOLD,
        silence_ms=DEFAULT_SILENCE_MS,
    ):
        self._probability = probability
        self._reset_vad = reset
        self.threshold = threshold
        self.negative_threshold = max(threshold - NEGATIVE_MARGIN, 0.01)
        self.silence_limit = max(1, int(silence_ms / 1000 / BLOCK_SECONDS))
        self.max_blocks = int(MAX_SEGMENT_SECONDS / BLOCK_SECONDS)
        self.min_voiced_blocks = max(1, int(MIN_SPEECH_SECONDS / BLOCK_SECONDS))
        self.tail_blocks = int(TAIL_SECONDS / BLOCK_SECONDS)
        self._pre_roll = collections.deque(
            maxlen=max(1, int(PRE_ROLL_SECONDS / BLOCK_SECONDS))
        )
        self.reset()

    @property
    def in_speech(self):
        """True si un segment est en cours (parole détectée, pas encore terminée)."""
        return self._in_speech

    def reset(self):
        """Oublie tout état en cours (et remet le VAD à zéro)."""
        self._pre_roll.clear()
        self._speech = []
        self._in_speech = False
        self._start_count = 0
        self._silence = 0
        self._voiced = 0
        if self._reset_vad is not None:
            self._reset_vad()

    def feed(self, block):
        """Traite un bloc. Retourne un segment (np.ndarray) quand il se termine."""
        probability = self._probability(block)

        if not self._in_speech:
            self._pre_roll.append(block)
            if probability >= self.threshold:
                self._start_count += 1
            else:
                self._start_count = 0

            if self._start_count >= START_BLOCKS:
                self._in_speech = True
                self._speech = list(self._pre_roll)  # inclut le pré-roll
                self._voiced = START_BLOCKS
                self._silence = 0
            return None

        self._speech.append(block)
        if probability < self.negative_threshold:
            self._silence += 1
        else:
            self._silence = 0
            self._voiced += 1

        if self._silence >= self.silence_limit or len(self._speech) >= self.max_blocks:
            return self._finish()
        return None

    def flush(self):
        """Termine de force le segment en cours (fin d'un fichier audio)."""
        if not self._in_speech:
            return None
        return self._finish()

    def _finish(self):
        keep = len(self._speech) - self._silence + min(self._silence, self.tail_blocks)
        segment = None
        if self._voiced >= self.min_voiced_blocks:
            segment = np.concatenate(self._speech[:keep])
        self.reset()
        return segment


def segment_audio(audio, probability, reset=None, **kwargs):
    """Rejoue un tableau audio (float32, 16 kHz) dans le même découpage que le micro.

    Permet de tester / évaluer le VAD sans micro. ``kwargs`` : threshold, silence_ms.
    """
    segmenter = Segmenter(probability, reset, **kwargs)
    segments = []
    for start in range(0, len(audio) - BLOCK_SIZE + 1, BLOCK_SIZE):
        segment = segmenter.feed(audio[start : start + BLOCK_SIZE])
        if segment is not None:
            segments.append(segment)
    last = segmenter.flush()
    if last is not None:
        segments.append(last)
    return segments
