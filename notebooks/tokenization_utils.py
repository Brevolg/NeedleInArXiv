# Исправленные функции токенизации для chunking.ipynb
import tiktoken

# Инициализация токенизатора
enc = tiktoken.get_encoding("cl100k_base")

def clean_text_for_tokenization(text: str) -> str:
    """Очищает текст от проблемных символов перед токенизацией"""
    if not isinstance(text, str):
        return ""
    # Заменяем или удаляем проблемные символы
    text = text.replace('\x00', '')  # Удаляем null байты
    return text

def n_tokens(text: str) -> int:
    """Подсчитывает количество токенов в тексте"""
    cleaned_text = clean_text_for_tokenization(text)
    return len(enc.encode(cleaned_text, allowed_special=set(), disallowed_special=()))

def fixed_chunk(text: str, chunk_size: int = 256, overlap: int = 32):
    """Фиксированный чанкинг с перекрытием"""
    # возвращает список (chunk_text, token_start, token_end)
    cleaned_text = clean_text_for_tokenization(text)
    tokens = enc.encode(cleaned_text, allowed_special=set(), disallowed_special=())
    if not tokens:
        return []
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        # Также игнорируем специальные токены при декодировании
        chunk_text = enc.decode(chunk_tokens, errors='ignore')
        chunks.append((chunk_text, start, end))
        if end == len(tokens):
            break
        start = end - overlap
    return chunks

def structure_aware_chunk(text: str, chunk_size: int = 256, overlap_units: int = 1, overlap_tokens: int = 32):
    """Структурно-зависимый чанкинг"""
    # Эта функция предполагает, что другие вспомогательные функции 
    # (split_into_units, pack_units) определены в основном ноутбуке
    pass  # Реализация зависит от контекста ноутбука