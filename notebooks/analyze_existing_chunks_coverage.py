# Анализ покрытия документов из существующих чанков в бенчмарке
from datasets import load_dataset
import pandas as pd
import numpy as np

def analyze_existing_chunks_coverage():
    """Анализирует, сколько документов из существующих чанков представлено в бенчмарке"""
    
    print("Загрузка данных...")
    # Загружаем вопросы
    ds_q = load_dataset("onyx-dot-app/EnterpriseRAG-Bench", "questions")
    df_questions = ds_q["test"].to_pandas()
    
    print(f"Загружено вопросов: {len(df_questions)}")
    
    # Загружаем чанки
    print("Загрузка чанков...")
    df_fixed_chunks = pd.read_parquet('notebooks/chunks_fixed_v1.parquet')
    df_structure_chunks = pd.read_parquet('notebooks/chunks_structure_v2.parquet')
    
    print(f"Загружено fixed чанков: {len(df_fixed_chunks)}")
    print(f"Загружено structure чанков: {len(df_structure_chunks)}")
    
    # Получаем уникальные doc_id из чанков
    fixed_doc_ids = set(df_fixed_chunks['doc_id'].unique())
    structure_doc_ids = set(df_structure_chunks['doc_id'].unique())
    
    print(f"Уникальных документов в fixed чанках: {len(fixed_doc_ids)}")
    print(f"Уникальных документов в structure чанках: {len(structure_doc_ids)}")
    
    # Проверяем, что множества документов совпадают (должны совпадать, так как созданы из одной выборки)
    if fixed_doc_ids == structure_doc_ids:
        print("Множества документов в обоих наборах чанков совпадают")
        sample_doc_ids = fixed_doc_ids
    else:
        print("Множества документов различаются, используем объединение")
        sample_doc_ids = fixed_doc_ids.union(structure_doc_ids)
    
    # Анализ структуры вопросов
    print("\nАнализ структуры вопросов...")
    print("Столбцы в df_questions:")
    for col in df_questions.columns:
        print(f"  {col}")
        
    # Проверяем наличие expected_doc_ids
    if 'expected_doc_ids' in df_questions.columns:
        print("\nНайден столбец 'expected_doc_ids' в вопросах")
        print("Анализируем документы, связанные с вопросами...")
        
        # Собираем все doc_id из expected_doc_ids
        all_expected_doc_ids = set()
        for doc_ids_list in df_questions['expected_doc_ids']:
            if isinstance(doc_ids_list, (list, np.ndarray)):
                for doc_id in doc_ids_list:
                    all_expected_doc_ids.add(doc_id)
            else:
                all_expected_doc_ids.add(doc_ids_list)
        
        print(f"Уникальных doc_id в expected_doc_ids: {len(all_expected_doc_ids)}")
        
        # Проверим покрытие
        benchmark_doc_ids = all_expected_doc_ids
        
        # Пересечение
        intersection = sample_doc_ids.intersection(benchmark_doc_ids)
        
        print(f"\nРезультаты анализа покрытия:")
        print(f"  Документов в чанках: {len(sample_doc_ids)}")
        print(f"  Документов в бенчмарке: {len(benchmark_doc_ids)}")
        print(f"  Документов из чанков, представленных в бенчмарке: {len(intersection)}")
        print(f"  Процент покрытия: {len(intersection) / len(sample_doc_ids) * 100:.2f}%")
        
        # Дополнительная информация
        # Подсчитываем, сколько вопросов относятся к документам из чанков
        questions_covering_sample = 0
        total_expected_doc_references = 0
        
        for doc_ids_list in df_questions['expected_doc_ids']:
            if isinstance(doc_ids_list, (list, np.ndarray)):
                for doc_id in doc_ids_list:
                    total_expected_doc_references += 1
                    if doc_id in sample_doc_ids:
                        questions_covering_sample += 1
            else:
                total_expected_doc_references += 1
                if doc_ids_list in sample_doc_ids:
                    questions_covering_sample += 1
        
        print(f"\nДополнительная информация:")
        print(f"  Всего ссылок на документы в вопросах: {total_expected_doc_references}")
        print(f"  Ссылок на документы из чанков: {questions_covering_sample}")
        print(f"  Процент ссылок на документы из чанков: {questions_covering_sample / total_expected_doc_references * 100:.2f}%")
        
        if len(intersection) > 0:
            print(f"  Среднее количество ссылок на один документ из чанков: {questions_covering_sample / len(intersection):.2f}")
        
        # Анализ по типам источников
        print(f"\nАнализ покрытия по типам источников:")
        coverage_by_source = []
        
        # Получаем типы источников для документов в чанках
        # Для этого нужно получить соответствия doc_id -> source_type
        doc_id_to_source_type = dict(zip(df_fixed_chunks['doc_id'], df_fixed_chunks['source_type']))
        
        # Группируем документы по типам источников
        docs_by_source = {}
        for doc_id in sample_doc_ids:
            source_type = doc_id_to_source_type.get(doc_id, "unknown")
            if source_type not in docs_by_source:
                docs_by_source[source_type] = set()
            docs_by_source[source_type].add(doc_id)
        
        for source_type, doc_ids_set in docs_by_source.items():
            sample_ids_of_type = doc_ids_set
            
            # Документы из бенчмарка данного типа
            benchmark_ids_of_type = set()
            for doc_id in doc_ids_set:
                if doc_id in benchmark_doc_ids:
                    benchmark_ids_of_type.add(doc_id)
            
            # Пересечение
            intersection_of_type = sample_ids_of_type.intersection(benchmark_ids_of_type)
            
            coverage_percent = len(intersection_of_type) / len(sample_ids_of_type) * 100 if sample_ids_of_type else 0
            
            coverage_by_source.append({
                'source_type': source_type,
                'sample_count': len(sample_ids_of_type),
                'benchmark_count': len(benchmark_ids_of_type),
                'intersection_count': len(intersection_of_type),
                'coverage_percent': coverage_percent
            })
        
        # Создаем DataFrame с результатами и сортируем по покрытию
        coverage_df = pd.DataFrame(coverage_by_source)
        coverage_df = coverage_df.sort_values('coverage_percent', ascending=False)
        
        print("\nПокрытие по типам источников:")
        for _, row in coverage_df.iterrows():
            print(f"  {row['source_type']}: {row['coverage_percent']:.2f}% "
                  f"({row['intersection_count']}/{row['sample_count']})")
        
        return {
            'total_sample_docs': len(sample_doc_ids),
            'docs_in_benchmark': len(intersection),
            'coverage_percent': len(intersection) / len(sample_doc_ids) * 100,
            'references_to_sample': questions_covering_sample,
            'total_references': total_expected_doc_references,
            'avg_references_per_doc': questions_covering_sample / len(intersection) if intersection else 0,
            'coverage_by_source': coverage_df
        }
    else:
        print("\nСтолбец 'expected_doc_ids' не найден в вопросах.")
        
        # Проверим все столбцы на наличие ID
        possible_id_columns = []
        for col in df_questions.columns:
            if 'id' in col.lower() or 'doc' in col.lower():
                possible_id_columns.append(col)
                print(f"  Возможное поле связи: {col}")
                try:
                    unique_count = df_questions[col].nunique()
                    print(f"    Уникальных значений: {unique_count}")
                except:
                    print(f"    Не удалось подсчитать уникальные значения (возможно, сложная структура данных)")
                
        return {
            'error': 'expected_doc_ids column not found',
            'possible_columns': possible_id_columns
        }

if __name__ == "__main__":
    result = analyze_existing_chunks_coverage()
    print("\nАнализ завершен.")