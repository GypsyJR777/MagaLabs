# Часть 1

import pandas as pd
import numpy as np
import seaborn as sns
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

pd.set_option('display.max_columns', None)

def createHeatMap(cols, df): 
    new_df = df[cols].fillna(df[cols].median())
    colors = ["#000080", "#0066cc", "#00b3ff", "#80ffff",
            "#ffffff", "#ff8080", "#ff0000", "#800000"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    plt.figure(figsize=(25, 20))
    corr_matrix = new_df.corr()
    heatmap = sns.heatmap(
        corr_matrix,
        cmap=cmap,                # Используем кастомную палитру
        annot=True,
        fmt=".2f",
        annot_kws={
            'size': 4,
            'weight': 'bold',
            'color': 'black'      # Белый текст для лучшей читаемости
        },
        linewidths=0.7,
        linecolor='black',        # Черные границы ячеек
        square=True,
        cbar_kws={
            'shrink': 0.8,
            'label': 'Уровень корреляции'
        },
        vmin=-1,
        vmax=1
    )
    plt.title('Корреляционная матрица с усиленными оттенками\n', 
            fontsize=22, pad=25, fontweight='bold')
    plt.xticks(
        rotation=45, 
        ha='right', 
        fontsize=6, 
        fontweight='bold'
    )
    plt.yticks(
        fontsize=6, 
        fontweight='bold'
    )
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Уровень корреляции', 
                fontsize=14, 
                fontweight='bold')
    plt.tight_layout()
    plt.show()

def baseStat(df):
    row = [df.nunique(), round(df.nunique() / len(df) * 100, 1),(df == 0).sum(axis=0), round((df == 0).sum(axis=0) / len(df) * 100, 1), df.isna().sum(), round(df.isna().sum() / len(df) * 100, 1), df.dtypes]
    return row

def moreStat(df):
    clear_df = df.dropna()
    row = [
        clear_df.count(),
        np.mean(clear_df), 
        np.median(clear_df),
        np.std(clear_df), 
        np.min(clear_df), 
        np.max(clear_df), 
        round((clear_df == 0).sum(axis=0) / len(clear_df) * 100, 1),
        round(df.isna().sum() / len(df) * 100, 1)
        ]
    return row

# Загрузка данных
df_original = pd.read_csv('HSE/BigData/HW1_var_8.csv', sep=';', encoding='utf-8')

df = df_original
# Убираем столбец варианта
df.pop('Номер варианта')
print(df.columns)


# Выделение числовых столбцов
true_numeric_cols = [
    'DTI', 'FULL_AGE_CHILD_NUMBER', 'DEPENDANT_NUMBER', 
    'BANKACCOUNT_FLAG', 'Period_at_work', 'age',
    'max90days', 'max60days', 'max30days', 'max21days', 
    'max14days', 'avg_num_delay', 'if_zalog',
    'num_AccountActive180', 'num_AccountActive90', 
    'num_AccountActive60', 'Active_to_All_prc',
    'numAccountActiveAll', 'numAccountClosed',
    'sum_of_paym_months', 'all_credits', 'Active_not_cc',
    'own_closed', 'min_MnthAfterLoan', 'max_MnthAfterLoan',
    'dlq_exist', 'thirty_in_a_year', 'sixty_in_a_year',
    'ninety_in_a_year', 'thirty_vintage', 'sixty_vintage',
    'ninety_vintage'
]
createHeatMap(true_numeric_cols, df)

# Рассчитаем кол-во уникальных, нулевых и пропущенных значений, а также их долю в % от общего кол-ва 
data = []
for column in df:
    data.append([column] + baseStat(df[column]))

print(tabulate(data, headers=['Column', 'Count Unique', '% of unique','Count Zeros', '% of zero', 'Count NaNs', '% of NaNs', 'data type'], tablefmt='orgtbl'))


# Поиск значения '*n.a.*', '', ' '
print('Проверяем возможные пустые значения')
maybe_nans = ['*n.a.*', '', ' ']
for column in df.columns:
    if (df[column].isin(maybe_nans).sum(axis=0) > 0):
        print(f"${column}: ${df[column].isin(maybe_nans).sum(axis=0)}")
        # df[column] = df[column].replace(maybe_nans, np.nan)


print('Вывод уникальных значений при их количестве меньше 10')
for column in df.columns:
    if (df[column].nunique() < 10):
        print(f"${column}: ${df[column].unique()}")

cat_rows = ['if_zalog',
    'dlq_exist', 'thirty_in_a_year', 'sixty_in_a_year',
    'ninety_in_a_year', 'thirty_vintage', 'sixty_vintage',
    'ninety_vintage'
    ]

# Рассчитаем среднее значение, медиану, стандартное отклонение, минимальное и максимальное значение
data = []
for column in df:
    if(df[column].dtypes != 'object' and column not in cat_rows):
        data.append([column] + moreStat(df[column]))

print(tabulate(
    data, 
    headers=['Column', 'count', 'mean', 'median','std','min','max', '% of zero', '% of NaNs'], 
    tablefmt='orgtbl'
    ))


# Действия по чистке датафрейма
# 1
df = df.drop('ID', axis=1)
# 2
df.replace('*n.a.*', np.nan, inplace=True)
# 3
df['EMPL_SIZE'].replace('>=100', '>100', inplace=True)
# 4
# print('Изменяем в столбце BANKACCOUNT_FLAG все значения больше 2 на 2')
df['BANKACCOUNT_FLAG'] = df['BANKACCOUNT_FLAG'].apply(
    lambda x: 2 if x >= 2 else x
)
# 5
original_rows = len(df)
df = df.dropna(subset=['EMPL_FORM', 'FAMILY_STATUS'])
deleted_rows = original_rows - len(df)
print(f"Удалено строк: {deleted_rows}")

original_rows = len(df)
df = df.dropna(subset=cat_rows)
deleted_rows = original_rows - len(df)
print(f"Удалено строк: {deleted_rows}")


data = []
for column in df:
    data.append([column] + baseStat(df[column]))

print(tabulate(data, headers=['Column', 'Count Unique', '% of unique','Count Zeros', '% of zero', 'Count NaNs', '% of NaNs', 'data type'], tablefmt='orgtbl'))


# Сравнение значений до и после изменений для EMPL_SIZE
# Создаем фигуру с двумя subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# График до изменений
ax1.pie(
    df_original['EMPL_SIZE'].value_counts(),
    labels=df_original['EMPL_SIZE'].value_counts().index,
    autopct='%1.1f%%',
    startangle=90,
    colors=['#66b3ff', '#99ff99', '#ffcc99'],
    wedgeprops={'edgecolor': 'black', 'linewidth': 0.5}
)
ax1.set_title('EMPL_SIZE до изменений', fontsize=14, pad=20)

# График после изменений
ax2.pie(
    df['EMPL_SIZE'].value_counts(),
    labels=df['EMPL_SIZE'].value_counts().index,
    autopct='%1.1f%%',
    startangle=90,
    colors=['#66b3ff', '#99ff99', '#ffcc99'],
    wedgeprops={'edgecolor': 'black', 'linewidth': 0.5}
)
ax2.set_title('EMPL_SIZE после изменений', fontsize=14, pad=20)

# Общий заголовок
plt.suptitle('Сравнение распределения EMPL_SIZE', fontsize=16, y=1.05)
plt.tight_layout()
plt.show()

# Удалим пропуски и создадим диаграмму для EDUCATION
df = df.dropna(subset=['EDUCATION'])

plt.figure(figsize=(10, 6))
sns.barplot(
    x=df['EDUCATION'].value_counts().values,
    y=df['EDUCATION'].value_counts().index,
    palette='viridis',
    edgecolor='black'
)

# Настройка осей и заголовка
plt.title('Распределение уровней образования', fontsize=16, pad=20)
plt.xlabel('Количество', fontsize=12)
plt.ylabel('Уровень образования', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Добавление значений на столбцы
for i, value in enumerate(df['EDUCATION'].value_counts().values):
    plt.text(value, i, f' {value}', va='center', fontsize=10)

plt.tight_layout()
plt.show()


# Удаляем пустую строку в EMPL_PROPERTY
df = df.dropna(subset=['EMPL_PROPERTY'])
plt.figure(figsize=(10, 6))
sns.barplot(
    x=df['EMPL_PROPERTY'].value_counts().values,
    y=df['EMPL_PROPERTY'].value_counts().index,
    palette='viridis',
    edgecolor='black'
)

# Настройка осей и заголовка
plt.title('Распределение cферы бизнеса работодателя', fontsize=16, pad=20)
plt.xlabel('Количество', fontsize=12)
plt.ylabel('Сфера бизнеса работодателя', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Добавление значений на столбцы
for i, value in enumerate(df['EMPL_PROPERTY'].value_counts().values):
    plt.text(value, i, f' {value}', va='center', fontsize=10)

plt.tight_layout()
plt.show()

print(f"mean(avg_num_delay): {np.mean(df['avg_num_delay'])}")

df['avg_num_delay'] = df['avg_num_delay'].replace(np.nan, np.mean(df['avg_num_delay']))


# Новая корреляционная матрица после изменений
new_cols = [
    'DTI', 'FULL_AGE_CHILD_NUMBER', 'DEPENDANT_NUMBER', 
    'Period_at_work', 'age',
    'max90days', 'max60days', 'max30days', 'max21days', 
    'max14days', 'avg_num_delay',
    'num_AccountActive180', 'num_AccountActive90', 
    'num_AccountActive60', 'Active_to_All_prc',
    'numAccountActiveAll', 'numAccountClosed',
    'sum_of_paym_months', 'all_credits', 'Active_not_cc',
    'own_closed', 'min_MnthAfterLoan', 'max_MnthAfterLoan'
]
createHeatMap(new_cols, df)

for col in df.columns:
    if col not in new_cols:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=df[col].value_counts().values,
            y=df[col].value_counts().index,
            palette='viridis',
            edgecolor='black'
        )
        # Настройка осей и заголовка
        plt.title(f'{col}', fontsize=16, pad=20)
        plt.xlabel('Количество', fontsize=12)
        plt.ylabel(f'{col}', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        # Добавление значений на столбцы
        for i, value in enumerate(df[col].value_counts().values):
            plt.text(value, i, f' {value}', va='center', fontsize=10)

        plt.tight_layout()
        plt.show()


for col in df.columns:
    print(f'{col} : {df[col].value_counts(dropna=False).head(1).index[0]}')


# Диаграмма возрастов
plt.figure(figsize=(10, 6))
sns.barplot(
            x=df['age'].value_counts().index,
            y=df['age'].value_counts().values,
            palette='viridis',
            edgecolor='black'
)
        # Настройка осей и заголовка
plt.title('age', fontsize=16, pad=20)
plt.xlabel('Количество', fontsize=12)
plt.ylabel('age', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
for i, value in enumerate(df['age'].value_counts().values):
    plt.text(value, i, value, va='center', fontsize=10)
plt.tight_layout()
plt.show()
print(f'Средний возраст: {np.mean(df['age'])}')


# Посмотрим на наш датафрейм
print(df)


# Перевод категориальных переменных в целочисленные
education_sex = ['SEX', 'EDUCATION']
data = []
for col in df.columns:
    if (col in education_sex):
        data = []
        for val in df[col].unique():
            if(val == 'Высшее/Второе высшее/Ученая степень'):
                continue
            data.append(val)
        for i in range(len(data)):
            print(f'{i} : {data[i]}')
            if(data[i] == 'высшее'):
                df.replace([data[i], 'Высшее/Второе высшее/Ученая степень'], i, inplace=True)
            df[data[i]]
