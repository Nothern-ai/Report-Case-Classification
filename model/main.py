import json
import torch

from dataset import predict_on_dataset, predict_single_text


def main(operation, file_path=None, text_input=None):
    model_path = 'bert_finetuned_model.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()

    with open('label_map.json', 'r', encoding='utf-8') as f:
        label_map = json.load(f)

    if operation == "predict_dataset":
        predict_on_dataset(model, file_path, label_map, device)
        print("Predictions saved to 'predicted_dataset.xlsx'")
    elif operation == "predict_text":
        label = predict_single_text(model, text_input, label_map, device)
        print(f"Predicted label: {label}")


def user_input():
    choice = input("Enter 1 to predict on a dataset file, 2 to input text directly: ")
    if choice == "1":
        file_path = input("Enter the path to the dataset file (xlsx): ")
        main("predict_dataset", file_path=file_path)
    elif choice == "2":
        text_input = input("Enter the text: ")
        main("predict_text", text_input=text_input)
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    user_input()
