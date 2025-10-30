import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1️⃣ Load dataset
df = pd.read_csv("colors.csv")

# 2️⃣ Buat kategori warna berdasarkan Hue
def categorize_color(row):
    r, g, b = row["Red (8 bit)"], row["Green (8 bit)"], row["Blue (8 bit)"]

    # Hitung nilai rata-rata (kecerahan) dan perbedaan antar kanal (kejenuhan)
    avg = (r + g + b) / 3
    diff = max(r, g, b) - min(r, g, b)

    # ---- Warna netral ----
    if avg < 40:
        return "Hitam"
    elif avg > 230 and diff < 25:
        return "Putih"
    elif diff < 20:
        return "Abu-Abu"

    # ---- Warna dasar ----
    dominant = np.argmax([r, g, b])

    # Merah dominan
    if dominant == 0:
        if g > b and r - g < 70:
            base = "Oranye"
        elif b > g and b > 100:
            base = "Magenta"
        else:
            base = "Merah"

    # Hijau dominan
    elif dominant == 1:
        if r > b and g - r < 60:
            base = "Kuning"
        elif b > r and b > 80:
            base = "Cyan"
        else:
            base = "Hijau"

    # Biru dominan
    else:
        if r > g and r > 100:
            base = "Ungu"
        elif g > r and g > 100:
            base = "Cyan"
        else:
            base = "Biru"

    # ---- Klasifikasi tingkat kecerahan ----
    if avg < 80:
        tone = "Gelap"
    elif avg > 200:
        tone = "Terang"
    elif avg > 150:
        tone = "Muda"
    else:
        tone = "Sedang"

    return f"{base} {tone}"



df["Color Category"] = df.apply(categorize_color, axis=1)


# 3️⃣ Pilih fitur dan target
X = df[["Red (8 bit)", "Green (8 bit)", "Blue (8 bit)"]]
y = df["Color Category"]

# 4️⃣ Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5️⃣ Buat dan latih model
model = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True))
model.fit(X_train, y_train)

# 6️⃣ Evaluasi model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"🎯 Akurasi Model: {accuracy * 100:.2f}%\n")
print("📊 Laporan Klasifikasi:")
print(classification_report(y_test, y_pred))

# 7️⃣ Simpan model
joblib.dump(model, "color_category_svm.pkl")
print("\n💾 Model disimpan ke 'color_category_svm.pkl'")

# 🔍 Contoh prediksi
for i in range(5):
    r, g, b = X_test.iloc[i]
    print(f"RGB({r},{g},{b}) → Prediksi: {y_pred[i]} | Asli: {y_test.iloc[i]}")




# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
# from sklearn.metrics import accuracy_score, classification_report
# import joblib
# from sklearn.ensemble import RandomForestClassifier

# # 1️⃣ Load dataset
# df = pd.read_csv("colors.csv")

# # 2️⃣ Buat kategori warna berdasarkan Hue
# def categorize_color(row):
#     r, g, b = row["Red (8 bit)"], row["Green (8 bit)"], row["Blue (8 bit)"]

#     # Hitung nilai rata-rata (kecerahan) dan perbedaan antar kanal (kejenuhan)
#     avg = (r + g + b) / 3
#     diff = max(r, g, b) - min(r, g, b)

#     # ---- Warna netral ----
#     if avg < 40:
#         return "Hitam"
#     elif avg > 230 and diff < 25:
#         return "Putih"
#     elif diff < 20:
#         return "Abu-Abu"

#     # ---- Warna dasar ----
#     dominant = np.argmax([r, g, b])

#     # Merah dominan
#     if dominant == 0:
#         if g > b and r - g < 70:
#             base = "Oranye"
#         elif b > g and b > 100:
#             base = "Magenta"
#         else:
#             base = "Merah"

#     # Hijau dominan
#     elif dominant == 1:
#         if r > b and g - r < 60:
#             base = "Kuning"
#         elif b > r and b > 80:
#             base = "Cyan"
#         else:
#             base = "Hijau"

#     # Biru dominan
#     else:
#         if r > g and r > 100:
#             base = "Ungu"
#         elif g > r and g > 100:
#             base = "Cyan"
#         else:
#             base = "Biru"

#     # ---- Klasifikasi tingkat kecerahan ----
#     if avg < 80:
#         tone = "Gelap"
#     elif avg > 200:
#         tone = "Terang"
#     elif avg > 150:
#         tone = "Muda"
#     else:
#         tone = "Sedang"

#     return f"{base} {tone}"



# df["Color Category"] = df.apply(categorize_color, axis=1)


# # 3️⃣ Pilih fitur dan target
# X = df[["Red (8 bit)", "Green (8 bit)", "Blue (8 bit)"]]
# y = df["Color Category"]

# # 4️⃣ Split dataset
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # 5️⃣ Buat dan latih model
# model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=300, random_state=42))
# model.fit(X_train, y_train)

# # 6️⃣ Evaluasi model
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)

# print(f"🎯 Akurasi Model: {accuracy * 100:.2f}%\n")
# print("📊 Laporan Klasifikasi:")
# print(classification_report(y_test, y_pred))

# # 7️⃣ Simpan model
# joblib.dump(model, "color_category_svm.pkl")
# print("\n💾 Model disimpan ke 'color_category_svm.pkl'")

# # 🔍 Contoh prediksi
# for i in range(5):
#     r, g, b = X_test.iloc[i]
#     print(f"RGB({r},{g},{b}) → Prediksi: {y_pred[i]} | Asli: {y_test.iloc[i]}")

