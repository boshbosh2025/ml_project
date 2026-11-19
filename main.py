import argparse
from model_pipeline import prepare_data, train_model, evaluate_model, save_model

def main():
    parser = argparse.ArgumentParser(description="Pipeline ML - OM5 Gradient Boosting")
    parser.add_argument('--data', type=str, required=True, help="Chemin du fichier CSV (ex: diabetic_data.csv)")
    parser.add_argument('--model_path', type=str, default='model.joblib', help="Chemin de sauvegarde du modèle")
    args = parser.parse_args()

    print("1. Préparation des données...")
    X_train, X_test, y_train, y_test = prepare_data(args.data)

    print("2. Entraînement du modèle (Gradient Boosting)...")
    model = train_model(X_train, y_train)

    print("3. Évaluation du modèle...")
    evaluate_model(model, X_test, y_test)

    print("4. Sauvegarde du modèle...")
    save_model(model, args.model_path)

if __name__ == "__main__":
    main()
