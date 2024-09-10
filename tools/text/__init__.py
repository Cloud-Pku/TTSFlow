from text.symbols import *
from text.symbol_table import SymbolTable
current_file_path = os.path.dirname(__file__)


_symbol_to_id = {s: i for i, s in enumerate(symbols)}

def cleaned_text_to_sequence(cleaned_text):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  phones = [_symbol_to_id[symbol] for symbol in cleaned_text]
  return phones

def cleaned_text_to_sequence_mfa(cleaned_text, en=False):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  if not en:
    unique_tokens = SymbolTable.from_file(os.path.join(current_file_path,"unique_text_tokens_v2punc.k2symbols")).symbols
  else:
    unique_tokens = SymbolTable.from_file(os.path.join(current_file_path,"unique_text_tokens_zh_en.k2symbols"))._sym2id
  token2idx = {token: idx for idx,
               token in enumerate(unique_tokens)}
  phones = [token2idx[token] for token in cleaned_text]
  # phones = [_symbol_to_id[symbol] for symbol in cleaned_text]
  return phones

