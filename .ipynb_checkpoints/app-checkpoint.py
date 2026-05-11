import pickle
import pandas as pd

# Cargar modelo
with open("modelo.pkl", "rb") as f:
    model = pickle.load(f)

print("=== Predicción de Resistencia del Concreto ===")

# Inputs
cement = float(input("Cemento: "))
slag = float(input("Escoria: "))
fly_ash = float(input("Ceniza volante: "))
water = float(input("Agua: "))
superplasticizer = float(input("Superplastificante: "))
coarse_agg = float(input("Agregado grueso: "))
fine_agg = float(input("Agregado fino: "))
age = float(input("Edad (días): "))

# Crear DataFrame con nombres correctos (esto elimina el warning)
data = pd.DataFrame([{
    "cement": cement,
    "slag": slag,
    "fly_ash": fly_ash,
    "water": water,
    "superplasticizer": superplasticizer,
    "coarse_agg": coarse_agg,
    "fine_agg": fine_agg,
    "age": age
}])

# Predicción
pred = model.predict(data)

print(f"\n🔹 Resistencia estimada: {pred[0]:.2f} MPa")