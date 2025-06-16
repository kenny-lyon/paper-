import pandas as pd
import numpy as np
import time
import optuna
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
import plotly.express as px
from tabulate import tabulate

# ------------------ 1. CARGA DE DATOS ------------------
ratings = pd.read_csv('rating.csv')
movies = pd.read_csv('movie.csv')

# Muestreo para pruebas rápidas (10%)
ratings_sample = ratings.sample(frac=0.1, random_state=42)
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings_sample[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# ------------------ 2. BO (Optuna) ------------------
tiempos_bo = []

def objective(trial):
    params = {
        'n_factors': trial.suggest_int('n_factors', 50, 200),
        'reg_all': trial.suggest_float('reg_all', 0.01, 0.2)
    }
    start = time.time()
    algo = SVD(**params, random_state=42)
    algo.fit(trainset)
    rmse = accuracy.rmse(algo.test(testset), verbose=False)
    tiempos_bo.append(time.time() - start)
    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=40, n_jobs=-1)

history_df = study.trials_dataframe()
bo_history = history_df['value'].cummin()

# ------------------ 3. Random Search ------------------
param_dist = {
    'n_factors': list(range(50, 201)),
    'reg_all': np.linspace(0.01, 0.2, 100)
}

def evaluate_svd(params):
    start = time.time()
    algo = SVD(**params, random_state=42)
    algo.fit(trainset)
    rmse = accuracy.rmse(algo.test(testset), verbose=False)
    return {'rmse': rmse, 'time': time.time() - start}

random_results = Parallel(n_jobs=-1)(delayed(evaluate_svd)({
    'n_factors': np.random.choice(param_dist['n_factors']),
    'reg_all': np.random.choice(param_dist['reg_all'])
}) for _ in range(40))

random_rmse = min([r['rmse'] for r in random_results])
random_time = sum([r['time'] for r in random_results]) / 60
random_history = pd.Series([r['rmse'] for r in random_results]).cummin()

# ------------------ 4. Grid Search ------------------
grid_params = {
    'n_factors': [50, 100, 150, 200],
    'reg_all': [0.01, 0.05, 0.1, 0.2]
}
grid_combinations = list(ParameterGrid(grid_params))[:40]  # Limitamos a 40 para igualdad
grid_results = [evaluate_svd(p) for p in grid_combinations]

grid_rmse = min([r['rmse'] for r in grid_results])
grid_time = sum([r['time'] for r in grid_results]) / 60
grid_history = pd.Series([r['rmse'] for r in grid_results]).cummin()

# ------------------ 5. Tabla de Resultados ------------------
resultados = pd.DataFrame({
    'Método': ['Bayesian Optimization', 'Random Search', 'Grid Search'],
    'RMSE': [study.best_value, random_rmse, grid_rmse],
    'Tiempo (min)': [sum(tiempos_bo)/60, random_time, grid_time],
    'Evaluaciones': [40, 40, 40]
})
print(tabulate(resultados, headers='keys', tablefmt='pretty'))

# ------------------ 6. Visualizaciones ------------------

# 1. Convergencia de BO
fig_convergencia = px.line(
    y=bo_history, x=range(1, 41),
    title="Convergencia de Bayesian Optimization",
    labels={'x': 'Iteración', 'y': 'RMSE'}
)

# 2. Comparación de RMSE
fig_rmse = px.bar(
    resultados, x='Método', y='RMSE',
    title='Comparación de RMSE por Método',
    color='Método', text='RMSE'
)

# 3. Comparación de Tiempos
fig_time = px.bar(
    resultados, x='Método', y='Tiempo (min)',
    title='Comparación de Tiempo por Método',
    color='Método', text='Tiempo (min)'
)

# 4. Exploración de Hiperparámetros
fig_hyperparams = px.parallel_coordinates(
    history_df[['params_n_factors', 'params_reg_all', 'value']],
    color='value', title='Exploración de Hiperparámetros (BO)',
    labels={'value': 'RMSE'}
)

# 5. Trade-off Tiempo vs RMSE
fig_tradeoff = px.scatter(
    resultados, x='Tiempo (min)', y='RMSE',
    color='Método', size='Evaluaciones',
    title='Trade-off: Tiempo vs RMSE'
)

# 6. BO vs Random vs Grid
max_len = max(len(bo_history), len(random_history), len(grid_history))
fig_all = px.line(
    pd.DataFrame({
        'Evaluación': list(range(max_len)),
        'BO': bo_history.reindex(range(max_len), method='ffill'),
        'Random Search': random_history.reindex(range(max_len), method='ffill'),
        'Grid Search': grid_history.reindex(range(max_len), method='ffill')
    }),
    x='Evaluación', y=['BO', 'Random Search', 'Grid Search'],
    title='Convergencia Comparativa: BO vs Random vs Grid',
    labels={'value': 'RMSE'}
)

# 7. BO vs Grid
fig_bo_vs_grid = px.line(
    pd.DataFrame({
        'Evaluación': range(40),
        'BO': bo_history,
        'Grid Search': grid_history
    }),
    x='Evaluación', y=['BO', 'Grid Search'],
    title='BO vs Grid Search',
    labels={'value': 'RMSE'}
)

# 8. BO vs Random
fig_bo_vs_random = px.line(
    pd.DataFrame({
        'Evaluación': range(40),
        'BO': bo_history,
        'Random Search': random_history
    }),
    x='Evaluación', y=['BO', 'Random Search'],
    title='BO vs Random Search',
    labels={'value': 'RMSE'}
)

# ------------------ 7. Mostrar Gráficos ------------------
fig_convergencia.show()
fig_rmse.show()
fig_time.show()
fig_hyperparams.show()
fig_tradeoff.show()
fig_all.show()
fig_bo_vs_grid.show()
fig_bo_vs_random.show()
