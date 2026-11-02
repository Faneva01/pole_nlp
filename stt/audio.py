"""Écoute du micro + détection de parole (Silero VAD).

``SpeechListener`` garde le flux micro ouvert entre deux messages (plus de
réouverture à chaque phrase) mais ne capte que pendant ``begin()`` … ``end()`` :
entre deux, les blocs sont jetés, ce qui évite que le robot ne s'entende lui-même
pendant que le TTS parle.

Les imports lourds (sounddevice, torch) sont faits à la demande pour que ce
module reste importable et testable sans matériel.
"""

import queue
import threading
import time

from .config import (
    BLOCK_SECONDS,
    BLOCK_SIZE,
    MAX_QUEUE_SECONDS,
    MAX_SEGMENT_SECONDS,
    SAMPLE_RATE,
)
from .segmenter import Segmenter


class SpeechListener:
    def __init__(self, vad_model, args, probability=None, clock=time.monotonic):
        if probability is None:
            from .optimization import speech_probability

            def probability(block):
                return speech_probability(vad_model, block)

        reset = getattr(vad_model, "reset_states", None)
        self.segmenter = Segmenter(
            probability,
            reset,
            threshold=args.vad_threshold,
            silence_ms=args.silence_ms,
        )
        self.mic = getattr(args, "mic", None)
        self.dropped_blocks = 0  # blocs perdus (file pleine) : à surveiller
        self._queue = queue.Queue(maxsize=int(MAX_QUEUE_SECONDS / BLOCK_SECONDS))
        self._listening = threading.Event()
        self._stream = None
        self._clock = clock

    # --- cycle de vie du flux micro ---------------------------------
    def start(self):
        if self._stream is not None:
            return
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=BLOCK_SIZE,
            device=self.mic,
            callback=self._on_audio,
        )
        self._stream.start()

    def close(self):
        self._listening.clear()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def _on_audio(self, indata, frames, time_info, status):
        # Appelée par sounddevice : aucun calcul ici, on met le bloc en file.
        if not self._listening.is_set():
            return
        try:
            self._queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            self.dropped_blocks += 1

    # --- fenêtre d'écoute -------------------------------------------
    def begin(self):
        """Commence à capter. Tout ce qui date d'avant est jeté."""
        self._listening.clear()
        self._flush_queue()
        self.segmenter.reset()
        self._listening.set()

    def end(self):
        """Arrête de capter (le robot parle, réfléchit...)."""
        self._listening.clear()

    def _flush_queue(self):
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _wait_block(self, timeout):
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # --- lecture des segments ---------------------------------------
    def next_segment(self, max_wait_seconds=None):
        """Attend le prochain segment de parole complet.

        ``max_wait_seconds`` borne l'attente du DÉBUT de la parole. Une phrase
        déjà commencée est toujours menée à son terme (elle n'est plus perdue
        quand le délai expire). Retourne None si aucune parole n'est apparue.
        """
        deadline = None if max_wait_seconds is None else self._clock() + max_wait_seconds
        while True:
            if (
                deadline is not None
                and not self.segmenter.in_speech
                and self._clock() >= deadline
            ):
                return None
            block = self._wait_block(0.25)
            if block is None:
                continue
            segment = self.segmenter.feed(block)
            if segment is not None:
                return segment

    def drain(self, grace_seconds=0.0):
        """Récupère les segments déjà captés pendant la transcription.

        Si la personne a repris la parole pendant que Whisper travaillait
        (pause en milieu de phrase), la suite est dans la file : on la renvoie
        au lieu de la jeter. Une parole en cours est attendue jusqu'à sa fin.
        ``grace_seconds`` ajoute une attente pour une suite pas encore commencée.
        """
        found = []
        end = self._clock() + grace_seconds
        hard_end = self._clock() + MAX_SEGMENT_SECONDS + 1.0
        while self._clock() < hard_end:
            try:
                block = self._queue.get_nowait()
            except queue.Empty:
                if self.segmenter.in_speech:
                    block = self._wait_block(0.25)
                elif self._clock() < end:
                    block = self._wait_block(0.05)
                else:
                    break
                if block is None:
                    continue
            segment = self.segmenter.feed(block)
            if segment is not None:
                found.append(segment)
                end = self._clock() + grace_seconds
        return found


def iter_speech_segments(vad_model, args, max_wait_seconds=None):
    """Génère les segments de parole détectés (interface d'origine conservée).

    ``max_wait_seconds`` rend l'attente finie ; None = écoute illimitée.
    """
    deadline = None if max_wait_seconds is None else time.monotonic() + max_wait_seconds
    with SpeechListener(vad_model, args) as listener:
        listener.begin()
        while True:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            segment = listener.next_segment(remaining)
            if segment is None:
                return
            yield segment


def listen_loop(vad_model, segments, args):
    """Écoute sans fin et met chaque segment de parole dans la file."""
    print("En écoute... parlez (Ctrl+C pour quitter).")
    for audio in iter_speech_segments(vad_model, args):
        segments.put(audio)
