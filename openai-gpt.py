import os
import openai
from datasets import load_dataset
import itertools
import random


def process_boolq(example):
    """Generate few shot examples."""
    question = example["question"]
    passage = example["passage"]
    answer = example["answer"]

    example_str = (
        "Question: "
        + question
        + "\nPassage: "
        + passage
        + "\nAnswer: "
        + ("Yes" if answer else "No")
        + "\n"
    )

    return {"example": example_str}


if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_API_KEY")

    boolq = load_dataset("boolq")

    few_shot_examples = boolq["train"].map(
        process_boolq, remove_columns=boolq["train"].column_names
    )
    few_shot_examples = [
        example["example"] for example in itertools.islice(few_shot_examples, 8)
    ]
    random.shuffle(few_shot_examples)
    few_shot_prompt = "".join(few_shot_examples)

    validation_examples = boolq["validation"].map(
        process_boolq, remove_columns=boolq["validation"].column_names
    )

    num_correct = 0
    for val_example in itertools.islice(validation_examples, 30):
        # remove answer
        [val_prompt, val_ground] = val_example["example"].rsplit(":", 1)
        val_ground = val_ground.strip()

        prompt = (
            "Answer with Yes and No for the following questions from the passages.\n"
            + few_shot_prompt
            + val_prompt
            + ":"
        )

        response = openai.Completion.create(
            model="davinci",
            prompt=prompt,
            temperature=0.7,
            max_tokens=1,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        answer = response["choices"][0]["text"].strip()

        if answer.strip() == val_ground.strip():
            num_correct += 1

        print(f"Ground Truth : {val_ground} || Response : {answer}\n")
    print(f"Accuracy = {num_correct} / 30 = {num_correct / 30.:.2f}")
