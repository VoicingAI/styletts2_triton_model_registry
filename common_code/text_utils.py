from nltk.tokenize import word_tokenize

# Copy of TextCleaner minimal from your meldataset.py
symbols = ["$"] + list(';:,.!?¡¿—…"«»“” ') + list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz') + \
          list("ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ")
dicts = {s: i for i, s in enumerate(symbols)}

from common_code.phonemizer_utils import PhonemizerManager

_english_phonemizer_mgr = PhonemizerManager(languages=["en"])

class TextCleaner:
    def __init__(self):
        self.word_index_dictionary = dicts
    def __call__(self, text):
        idxs = []
        for ch in text:
            if ch in self.word_index_dictionary:
                idxs.append(self.word_index_dictionary[ch])
        return idxs

def english_phonemizer():
    backend = _english_phonemizer_mgr.get_backend("en")
    if backend is None:
        raise RuntimeError("English phonemizer backend is unavailable.")
    return backend.phonemize
