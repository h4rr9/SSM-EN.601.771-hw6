import os
from datasets import load_dataset
import itertools
import json
import requests
import random

def process_boolq(example):
    """Generate few shot examples."""
    question = example["question"]
    passage = example["passage"]
    answer = example["answer"]

    example_str = (
        "\nPassage: "
        + passage
        + "Question: "
        + question
        + "\nAnswer: "
        + ("Yes" if answer else "No")
        + "\n"
    )

    return {"example": example_str}

if __name__ == "__main__":
    API_TOKEN = os.getenv("HF_API_KEY")
    headers = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type" : "application/json"}
    API_URL = "https://api-inference.huggingface.co/models/bigscience/bloomz"

    def query(payload):
        """Query result from api endpoint."""
        data = json.dumps(payload)
        response = requests.request("POST", API_URL, headers=headers, data=data)
        return json.loads(response.content.decode("utf-8"))

    boolq = load_dataset("boolq")

    few_shot_examples = boolq["train"].map(
        process_boolq
    )

    few_shot_examples = [
        example for example in few_shot_examples[[0, 1, 6, 7]]['example']
    ]
    random.shuffle(few_shot_examples)
    few_shot_prompt = "".join(few_shot_examples)

    validation_examples = boolq["validation"].map(
        process_boolq, remove_columns=boolq["validation"].column_names
    )
    num_correct = 0
    for val_example in itertools.islice(validation_examples, 101):
        [val_prompt, val_ground] = val_example["example"].rsplit(":", 1)
        val_ground = val_ground.strip()

        prompt = (
            "Answer the Question with Yes or No based on the context in the Paragraph.\n"
            + few_shot_prompt
            + val_prompt
            + ":"
        )
        payload = {"inputs": prompt}
        response = query(payload)
        try:
            [model_response] = response
            [_, answer] = model_response['generated_text'].rsplit(':', 1)
            answer = answer.strip()

            if answer.strip() == val_ground:
                num_correct += 1
        except ValueError:
            pass


        print(f"Ground Truth : {val_ground} || Response : {answer}\n")
    print(f"Accuracy = {num_correct} / 100 = {num_correct / 100.:.2f}")
