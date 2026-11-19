"""LSTM Model Training Script.

Trains the causal 2-layer LSTM sequence classifier with Focal Loss,
class balancing weights, and early stopping.
"""

import argparse
import os
import numpy as np

from pose_control.config import (
    NUM_CLASSES,
    SEQUENCE_LENGTH,
    TOTAL_FEATURE_DIM,
    CLASS_NAMES,
    ActionClass,
)
from pose_control.dataset.generator import DatasetBuilder
from pose_control.model.network import (
    create_combat_lstm_model,
    MultiClassFocalLoss,
    HAS_TENSORFLOW,
)

if HAS_TENSORFLOW:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from sklearn.metrics import classification_report, confusion_matrix


def train_model(
    dataset_path: str = "dataset/dataset_splits.npz",
    epochs: int = 50,
    batch_size: int = 32,
    output_dir: str = "saved_models",
):
    """Trains the LSTM sequence model on gesture dataset."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load or Generate Dataset
    if not os.path.exists(dataset_path):
        print(f"[Dataset] No dataset found at {dataset_path}. Generating synthetic dataset...")
        builder = DatasetBuilder(output_dir="dataset")
        splits = builder.generate_balanced_dataset(samples_per_class=150)
        X_train, y_train = splits["train"]
        X_val, y_val = splits["val"]
        X_test, y_test = splits["test"]
    else:
        print(f"[Dataset] Loading dataset from {dataset_path}...")
        data = np.load(dataset_path)
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
        X_test, y_test = data["X_test"], data["y_test"]

    print(f"Train Shape: {X_train.shape}, Val Shape: {X_val.shape}, Test Shape: {X_test.shape}")

    # One-hot encode targets for training
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=NUM_CLASSES)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)

    # 2. Compute Cost-Sensitive Class Weights
    class_counts = np.bincount(y_train, minlength=NUM_CLASSES)
    total_samples = len(y_train)
    class_weights = total_samples / (NUM_CLASSES * np.maximum(class_counts, 1).astype(np.float32))
    print(f"Computed Class Weights: {dict(zip([c.name for c in ActionClass], class_weights.round(3)))}")

    # 3. Instantiate Model
    print("[Model] Building Causal 2-Layer LSTM Model...")
    model = create_combat_lstm_model(
        sequence_length=SEQUENCE_LENGTH,
        feature_dim=TOTAL_FEATURE_DIM,
        num_classes=NUM_CLASSES,
    )
    model.summary()

    # 4. Compile with Multi-Class Focal Loss & Adam
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    loss_fn = MultiClassFocalLoss(gamma=2.0, class_weights=tf.constant(class_weights, dtype=tf.float32))

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"],
    )

    # 5. Callbacks
    best_model_path = os.path.join(output_dir, "combat_lstm_best.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1),
        ModelCheckpoint(best_model_path, monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    # 6. Fit Model
    print(f"[Training] Beginning training for {epochs} epochs...")
    history = model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # 7. Evaluate on Test Split
    print("[Evaluation] Evaluating on held-out test split...")
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc * 100:.2f}%")

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=-1)

    target_names = [CLASS_NAMES[c] for c in ActionClass]
    report = classification_report(y_test, y_pred, target_names=target_names)
    print("\nClassification Report:\n" + report)

    # Save final model
    final_path = os.path.join(output_dir, "combat_lstm_final.keras")
    model.save(final_path)
    print(f"Saved trained model to {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train LSTM Gesture Classifier")
    parser.add_argument("--dataset", type=str, default="dataset/dataset_splits.npz")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=str, default="saved_models")
    args = parser.parse_args()

    train_model(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
