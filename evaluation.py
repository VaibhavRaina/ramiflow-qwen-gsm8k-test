from benchmark_utils import extract_predicted_answer, extract_reference_answer


def score_predictions(
    rows: list[dict[str, str]], outputs: list[str],
) -> dict[str, object]:
    if len(rows) != len(outputs):
        raise ValueError("prediction count must match evaluation rows")
    examples: list[dict[str, object]] = []
    correct_count = 0
    for index, (row, output) in enumerate(zip(rows, outputs, strict=True)):
        reference = extract_reference_answer(row["answer"])
        prediction = extract_predicted_answer(output)
        correct = prediction == reference
        correct_count += int(correct)
        examples.append({
            "index": index,
            "question": row["question"],
            "reference_answer": reference,
            "prediction": prediction,
            "raw_output": output,
            "correct": correct,
        })
    accuracy = correct_count / len(rows) if rows else 0.0
    return {"metrics": {"gsm8k_accuracy": accuracy}, "examples": examples}
