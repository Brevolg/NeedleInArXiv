# Фильтрация вопросов, относящихся к документам из нашей выборки
from datasets import load_dataset
import pandas as pd
import numpy as np

def filter_questions_for_sample():
    """Фильтрует вопросы, относящиеся к документам из нашей выборки, и сохраняет их в паркетник"""
    
    print("Загрузка данных...")
    # Загружаем вопросы
    ds_q = load_dataset("onyx-dot-app/EnterpriseRAG-Bench", "questions")
    df_questions = ds_q["test"].to_pandas()
    
    print(f"Загружено вопросов: {len(df_questions)}")
    
    # Загружаем чанки для получения списка doc_id из нашей выборки
    print("Загрузка чанков для получения списка документов...")
    df_fixed_chunks = pd.read_parquet('notebooks/chunks_fixed_v1.parquet')
    
    # Получаем уникальные doc_id из чанков
    sample_doc_ids = set(df_fixed_chunks['doc_id'].unique())
    print(f"Уникальных документов в чанках: {len(sample_doc_ids)}")
    
    # Фильтруем вопросы, которые ссылаются на документы из нашей выборки
    print("Фильтрация вопросов...")
    
    # Создаем маску для фильтрации
    mask = []
    total_references_to_sample = 0
    
    for idx, row in df_questions.iterrows():
        expected_doc_ids = row['expected_doc_ids']
        # Проверяем, является ли expected_doc_ids списком или массивом
        if isinstance(expected_doc_ids, (list, np.ndarray)):
            # Проверяем, есть ли хотя бы один документ из expected_doc_ids в нашей выборке
            has_matching_doc = any(doc_id in sample_doc_ids for doc_id in expected_doc_ids)
            if has_matching_doc:
                total_references_to_sample += sum(1 for doc_id in expected_doc_ids if doc_id in sample_doc_ids)
        else:
            # expected_doc_ids - скалярное значение
            has_matching_doc = expected_doc_ids in sample_doc_ids
            if has_matching_doc:
                total_references_to_sample += 1
                
        mask.append(has_matching_doc)
    
    # Фильтруем DataFrame
    filtered_questions = df_questions[mask]
    
    print(f"Отфильтровано вопросов, относящихся к документам из выборки: {len(filtered_questions)}")
    print(f"Всего ссылок на документы из выборки: {total_references_to_sample}")
    
    # Сохраняем отфильтрованные вопросы в паркетный файл
    output_file = 'notebooks/questions_for_sample.parquet'
    filtered_questions.to_parquet(output_file, index=False)
    print(f"Отфильтрованные вопросы сохранены в {output_file}")
    
    # Выводим информацию о сохраненных вопросах
    print("\nИнформация о сохраненных вопросах:")
    print(f"  Количество вопросов: {len(filtered_questions)}")
    print(f"  Столбцы: {list(filtered_questions.columns)}")
    
    # Пример нескольких вопросов
    print("\nПримеры первых 3 вопросов:")
    for i in range(min(3, len(filtered_questions))):
        print(f"  Вопрос {i+1}:")
        print(f"    ID: {filtered_questions.iloc[i]['question_id']}")
        print(f"    Тип: {filtered_questions.iloc[i]['question_type']}")
        print(f"    Вопрос: {filtered_questions.iloc[i]['question'][:100]}...")
        print(f"    Ожидаемые doc_id: {filtered_questions.iloc[i]['expected_doc_ids']}")
        print()

if __name__ == "__main__":
    filter_questions_for_sample()
    print("Фильтрация завершена.")