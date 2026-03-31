"""Simple manual entry point for testing sentiment predictions."""

from __future__ import annotations

from models.sentiment.predict import predict_sentiment


EXAMPLE_TEXTS = [
    "I love the new iPhone camera. Apple really nailed it this year.",
    "The customer support for this product was terrible and frustrating.",
    "Google announced another update today.",
]


def main() -> None:
    """Run a few sample predictions with the trained sentiment model."""

    for text in EXAMPLE_TEXTS:
        result = predict_sentiment(text)
        print("=" * 80)
        print(f"Input: {result['input_text']}")
        print(f"Cleaned: {result['cleaned_text']}")

        if result["error"] is not None:
            print(f"Error: {result['error']}")
            continue

        print(f"Predicted label: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("Class probabilities:")
        for label, probability in result["class_probabilities"].items():
            print(f"  - {label}: {probability:.4f}")


if __name__ == "__main__":
    main()
