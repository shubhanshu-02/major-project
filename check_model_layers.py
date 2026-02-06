from keras.models import load_model

model = load_model('model.h5')

print("MODEL ARCHITECTURE SUMMARY:")
print("=" * 50)
model.summary()

print(f"\nTOTAL LAYERS: {len(model.layers)}")
print(f"TRAINABLE PARAMETERS: {model.count_params():,}")

print("\nLAYER BREAKDOWN:")
print("-" * 30)
for i, layer in enumerate(model.layers):
    print(f"{i+1:2d}. {layer.__class__.__name__:15s} - {layer.name}")
    if hasattr(layer, 'filters'):
        print(f"    Filters: {layer.filters}, Kernel: {layer.kernel_size}")
    elif hasattr(layer, 'units'):
        print(f"    Units: {layer.units}")