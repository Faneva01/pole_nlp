"""Module speech-to-text : Whisper + Silero VAD.

Les noms publics sont chargés à la demande : ``import stt.segmenter`` (ou
``stt.config``) ne charge ni torch ni Whisper, ce qui permet de tester le
découpage de la parole sans matériel ni modèle.
"""

_EXPORTS = {
    "build_args": "config",
    "change_model": "transcriber",
    "close_stt": "transcriber",
    "load_stt": "transcriber",
    "transcribe_message": "transcriber",
    "transcribe_message_info": "transcriber",
    "transcrire_message": "transcriber",
    "transcrire_message_info": "transcriber",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        import importlib

        module = importlib.import_module("." + _EXPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError("module 'stt' has no attribute %r" % name)
