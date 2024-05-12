import os
import pandas as pd
from pandas.plotting import scatter_matrix
import plotly.express as px

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('//wsl.localhost/Ubuntu/var/www/html/faculdade/tec_challenge_1/insurance.csv')

# print(dataset.head()) # Exemplo dos registros
# print(dataset.shape) # Análise de dimensão, número de registros X colunas
# print(dataset.info()) # Análise dos tipos de informações

# Gráficos gerais utilizando o parametro de se fuma ou não como base
# fig = px.scatter_matrix(
#     dataset,
#     dimensions=["age", "bmi", "children", "charges"],
#     color="smoker",
#     color_discrete_map={"yes": "red", "no": "blue"}  # Mapeamento de cores
# )
# fig.update_traces(diagonal_visible=False)  # Não mostrar distribuição em diagonal
# fig.show()


# Gráfico para exibir a média de regiões com valores mais altos
# average_charges_by_region = dataset.groupby('region')['charges'].mean().reset_index()
# region_with_max_average_charges = average_charges_by_region.loc[average_charges_by_region['charges'].idxmax()]['region']

# fig = px.bar_polar(
#     average_charges_by_region,
#     r="charges",
#     theta="region",
#     color="region",
#     template="plotly_dark",
#     color_discrete_sequence=px.colors.sequential.Plasma_r,
#     title=f"Região com a maior média de cobranças: {region_with_max_average_charges}"
# )
# fig.show()


# # Gráfico de dispersão interativo com opção de diferenciar fumantes e não fumantes
# fig = px.scatter(
#     dataset,
#     x="bmi",
#     y="charges",
#     color="smoker",
#     trendline="ols",  # Adiciona uma linha de tendência (regressão linear)
#     category_orders={"smoker": ["yes", "no"]},  # Define a ordem das categorias para a legenda
#     title="Relação entre BMI e Charges",
#     labels={"bmi": "BMI", "charges": "Charges", "smoker": "Fumante"}
# )

# # Adiciona botões de alternância para exibir/ocultar os pontos de fumantes e não fumantes
# fig.update_layout(
#     updatemenus=[
#         dict(
#             buttons=list([
#                 dict(label="Ambos",
#                      method="update",
#                      args=[{"visible": [True, True, True, True]}, {"title": "Relação entre BMI e Charges"}]),
#                 dict(label="Apenas Fumantes",
#                      method="update",
#                      args=[{"visible": [True, True, False, False]}, {"title": "Relação entre BMI e Charges - Apenas Fumantes"}]),
#                 dict(label="Apenas Não Fumantes",
#                      method="update",
#                      args=[{"visible": [False, False, True, True]}, {"title": "Relação entre BMI e Charges - Apenas Não Fumantes"}])
#             ]),
#             direction="down",
#             showactive=True,
#             x=1.1,
#             xanchor="left",
#             y=1.1,
#             yanchor="top"
#         ),
#     ]
# )

# # Ajusta o tamanho dos marcadores
# fig.update_traces(marker=dict(size=8))

# fig.show()

# Gráfico de dispersão Idade X Valores
# fig = px.scatter(
#     dataset,
#     x="age",
#     y="charges",
#     title="Relação entre Idade e Charges",
#     labels={"age": "Idade", "charges": "Charges", "age_category": "Categoria de BMI"}
# )

# fig.show()

# Visualizar valores que estão faltando(colunas null)
# print(dataset.info())

# set(dataset['region'])
# print(dataset['region'].value_counts())

df_train, df_test = train_test_split(dataset, test_size=0.2, random_state=7)
# print(len(df_train), "treinamento +", len(df_test), "teste")

# Verificando a correlação dos valores de colunas em relação ao 'charges'
numeric_data = dataset.select_dtypes(include=[np.number])
corr_matrix = numeric_data.corr()
# print(corr_matrix['charges'].sort_values(ascending=False))


# Tratativa para separar os registros fumantes e não fumantes para as bases
dataset['sex'] = dataset['sex'].map({'male': 1, 'female': 0})
dataset['smoker'] = dataset['smoker'].map({'yes': 1, 'no': 0})

dataset['smoker_cat'] = np.where(dataset['smoker'] == 1, 'smoker', 'not-smoker')

# Resultado total fumantes/nao fumantes
# print(dataset['smoker_cat'].value_counts())

# Tratativa para separar os registros por idade igualmente entre as bases
dataset['age_cat'] = np.ceil(dataset['age'] / 1.5)
dataset['age_cat'].where(dataset['age_cat'] < 50, 50.0, inplace=True)

dataset['age_cat'] = pd.cut(dataset['age'],
                               bins=[0, 18, 30, 40, 50, 60, np.inf],
                               labels=['0-18', '19-30', '31-40', '41-50', '51-60', '60+'])

# print(dataset['age_cat'].value_counts())

# Criando variáveis de base de teste e treino com média de dados do 'age' e 'smoker' parecidas
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(dataset, dataset[['smoker_cat', 'age_cat']]):
    strat_train_set = dataset.loc[train_index]
    strat_test_set = dataset.loc[test_index]

# Verificando se foi distribuido corretamente os fumantes entre as bases
# print(strat_test_set['smoker_cat'].value_counts() / len(strat_test_set))
# print(strat_train_set['smoker_cat'].value_counts() / len(strat_train_set))
# print(dataset['smoker_cat'].value_counts() / len(dataset))

# Verificando se foi distribuido corretamente as ages entre as bases
# print(strat_test_set['age_cat'].value_counts() / len(strat_test_set))
# print(strat_train_set['age_cat'].value_counts() / len(strat_train_set))
# print(dataset['age_cat'].value_counts() / len(dataset))

for set_ in (strat_train_set, strat_test_set):
    set_.drop('smoker_cat', axis= 1, inplace=True)
    set_.drop('age_cat', axis= 1, inplace=True)


train_data = strat_train_set.copy()

dataset_num = train_data.drop(columns=['region'])

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")), # Substituindo valores nulos pela mediana
    ('std_scaler', StandardScaler()), # Padronizando a escala dos dados
])

dataset_num_tr = num_pipeline.fit_transform(dataset_num)

num_attribs = list(dataset_num)
cat_attribs = ['region']

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(), cat_attribs)
])

train_prepared = full_pipeline.fit_transform(train_data)

print(train_prepared.shape)

column_names = [
    'age', 'sex', 'bmi', 'children', 'smoker', 'charges', 'northwest', 'northeast', 'southwest', 'southeast'
]

train_df = pd.DataFrame(data=train_prepared, columns=column_names)

print(train_df.head())

train_labels = strat_train_set['charges'].copy()

model_dtr = DecisionTreeRegressor(max_depth=10)
model_dtr.fit(train_prepared, train_labels)

some_data = train_data.iloc[:5]
some_labels = train_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
predictions = model_dtr.predict(some_data_prepared)


print("Predictions: ", model_dtr.predict(some_data_prepared))
print("Labels: ", list(some_labels))


# Erro ao quadrado
train_predictions = model_dtr.predict(train_prepared)
lin_mse = mean_squared_error(train_labels, train_predictions)
lin_rmse = np.sqrt(lin_mse)
print("Erro ao Quadrado: ",lin_rmse)

# Erro absoluto
lin_mae = mean_absolute_error(train_labels, train_predictions)
print("Erro Absoluto: ",lin_mae)

r2 = r2_score(train_labels, train_predictions)
print("r²", r2)

plt.show()

def calculate_mape(labels, predictions):
    errors = np.abs(labels - predictions)
    relative_errors = errors / np.abs(labels)
    mape = np.mean(relative_errors) * 100
    return mape

mape_result = calculate_mape(train_labels, train_predictions)

print(f"O MAPE é: {mape_result:.2f}%")

# Visualizar valores que estão faltando(colunas null)
# print(dataset.info())

# print(dataset.head())