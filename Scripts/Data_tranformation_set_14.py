import pandas as pd
import re

# Leer archivo de texto crudo
with open("TFT_set_14_raw.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Quitar saltos de línea y separar por tabuladores o espacios múltiples
rows = [re.split(r"\s{2,}|\t", line.strip()) for line in lines if line.strip()]

# Crear DataFrame crudo
df = pd.DataFrame(rows[1:], columns=rows[0])

# Normalizar columnas
df["Cost"] = df["Cost"].str.replace("💰", "").astype(int)
df["Star"] = df["Star"].str.count("⭐")  # ⭐⭐ -> 2
df["Range"] = df["Range"].astype(str).str.extract(r"(\d+)").astype(float)

# Convertir numéricas
for col in ["HP", "AD", "AS", "AR", "MR", "SMP", "MP"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Agrupar por campeón
resultados = []
for champ, grupo in df.groupby("Champions"):
    grupo = grupo.sort_values("Star")

    health = grupo["HP"].tolist()
    attack = grupo["AD"].tolist()

    inicial_mana = grupo["SMP"].iloc[0] if "SMP" in grupo else 0
    skill_cost  = grupo["MP"].iloc[0] if "MP" in grupo else 0
    attack_speed = grupo["AS"].iloc[0]
    armor = grupo["AR"].iloc[0]
    mr = grupo["MR"].iloc[0]
    attack_range = grupo["Range"].iloc[0]
    cost = grupo["Cost"].iloc[0]

    fila = {
        "health1": health[0] if len(health) > 0 else "",
        "health2": health[1] if len(health) > 1 else "",
        "health3": health[2] if len(health) > 2 else "",
        "inicial_mana": inicial_mana,
        "skill_cost": skill_cost,
        "attack1": attack[0] if len(attack) > 0 else "",
        "attack2": attack[1] if len(attack) > 1 else "",
        "attack3": attack[2] if len(attack) > 2 else "",
        "attack_speed": attack_speed,
        "armor": armor,
        "mr": mr,
        "attack_range": attack_range,
        "cost": cost
    }
    resultados.append(fila)

# Guardar en CSV
final_df = pd.DataFrame(resultados)
final_df.to_csv("TFT_set_14.csv", index=False)

print("✅ Archivo generado: tft_champions_clean.csv")
