import argparse
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from utils.label_utils import save_label_map

def build_and_train(data_dir='dataset', img_size=160, batch_size=32, epochs=12, model_out='models/signclassifier.h5'):
    data_dir = Path(data_dir)

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_gen = datagen.flow_from_directory(
        str(data_dir),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        subset='training',
        class_mode='categorical'
    )

    val_gen = datagen.flow_from_directory(
        str(data_dir),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        subset='validation',
        class_mode='categorical'
    )

    num_classes = len(train_gen.class_indices)
    print("Class indices:", train_gen.class_indices)

    save_label_map({v:k for k,v in train_gen.class_indices.items()}, Path(model_out).parent / 'labels.json')

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size,img_size,3))
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    model.fit(train_gen, validation_data=val_gen, epochs=epochs)

    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_out)
    print(f"✅ Model saved to {model_out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='dataset')
    parser.add_argument('--img_size', type=int, default=160)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--model_out', default='models/signclassifier.h5')
    args = parser.parse_args()
    build_and_train(data_dir=args.data_dir, img_size=args.img_size,
                    batch_size=args.batch, epochs=args.epochs, model_out=args.model_out)
