from tensorflow.keras.models import load_model

# Load your old H5 model
model = load_model("mv2_tobacco_model.h5", compile=False)

# Save in TF 2.20 native format (.keras)
model.save("mv2_tobacco_model.keras")

print("✅ Conversion complete: mv2_tobacco_model.keras created")
