import argparse
import torch
import othello
from policy_function import Policy
from value_function import Value

def load_best_models(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    best_policy_state = checkpoint["best_policy_state"]
    best_value_state = checkpoint["best_value_state"]
    return best_policy_state, best_value_state

def export_to_onnx(model, dummy_input, onnx_file, input_names=["input"], output_names=["output"]):
    # Export the model to ONNX format.
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file,
        input_names=input_names,
        output_names=output_names,
        opset_version=11
    )
    print(f"Exported {onnx_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Load best policy and value models from a checkpoint and export them to ONNX."
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to the checkpoint file (e.g., checkpoint.pth)"
    )
    args = parser.parse_args()
    
    checkpoint_path = args.checkpoint_path

    # Load the best model states from the checkpoint
    best_policy_state, best_value_state = load_best_models(checkpoint_path)

    # Initialize model instances using the board size from othello
    board_input_size = othello.BOARD_SIZE ** 2
    policy_model = Policy(board_input_size)
    value_model = Value(board_input_size)

    # Load the best state dictionaries into the models
    policy_model.load_state_dict(best_policy_state)
    value_model.load_state_dict(best_value_state)

    # Set models to evaluation mode
    policy_model.eval()
    value_model.eval()

    # Create a dummy input tensor (adjust shape if needed)
    dummy_input = torch.randn(1, board_input_size)

    # Export the models to ONNX format
    export_to_onnx(policy_model, dummy_input, "best_policy.onnx", input_names=["input"], output_names=["policy_output"])
    export_to_onnx(value_model, dummy_input, "best_value.onnx", input_names=["input"], output_names=["value_output"])

if __name__ == "__main__":
    main()
