import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split  
import optuna

# Cargar datos
ratings = pd.read_csv('rating.csv')  # 20M filas
movies = pd.read_csv('movie.csv')    # 27k filas (opcional para análisis)

# Muestreo aleatorio (para acelerar la optimización)
ratings_sample = ratings.sample(frac=0.02, random_state=42)  # 2% ~ 400mil ratings

# Configurar Surprise
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings_sample[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

def objective(trial):
    # Hiperparámetros a optimizar
    n_factors = trial.suggest_int('n_factors', 50, 200)
    reg_all = trial.suggest_float('reg_all', 0.01, 0.2)
    
    # Entrenar SVD
    algo = SVD(n_factors=n_factors, reg_all=reg_all, random_state=42)
    algo.fit(trainset)
    
    # Evaluar RMSE
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    return rmse

# Ejecutar la optimización
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=40)

# Resultados finales
print("\n--- Resultados de la Optimización ---")
print(f"Mejor RMSE: {study.best_value:.4f}")
print(f"Mejores parámetros: {study.best_params}")

# Gráfico de convergencia de BO
optuna.visualization.plot_optimization_history(study).show()