import os
from openai import OpenAI


def create_evaluation_prompt(context, question, answer_1, answer_2):
    """Create the evaluation prompt for GPT."""
    return [
        {'role': 'user', 'content': f'''You are an evaluator assessing the quality of an AI-generated response based on a provided human-labeled context and a human-written ground-truth response.

### Context:
{context}

### Question:
{question}

### Answer_1 (Ground Truth Answer):
{answer_1}

### Answer_2 (Model-generated Answer):
{answer_2}

### Task:
Compare Answer_2 with Answer_1 using the provided Context and Question. Focus on the following aspects:
- Helpfulness: Does the answer provide useful and relevant information?
- Relevance: Does it stay on topic and align with the context?
- Accuracy: Does it reflect the facts stated in the context?
- Level of Detail: Is the response thorough and precise?

Then, do the following:
1. Provide a brief explanation comparing Answer_2 with Answer_1.
2. Give a numerical score (0 to 10) for Answer_2 based on its quality relative to Answer_1.
3. Do not score Answer_1. Use it only as the gold reference.

### Output Format:
Explanation: <Your reasoning here>  
Score: <A number from 0 to 10>'''}
    ]



def extract_score(evaluation_text):
    """Extract numerical score from GPT evaluation text."""
    if not evaluation_text:
        return None
    
    try:
        # Replace newlines with spaces to handle multi-line formats
        evaluation_text = evaluation_text.replace('\n', ' ')
        
        # Handle both regular "Score:" and markdown formatted "**Score:**"
        if "Score:" in evaluation_text:
            score_pattern = "Score:"
        else:
            return None
        
        # Find the score part
        score_part = evaluation_text.split(score_pattern)[1].strip()
        
        # Remove any non-numeric characters except decimal point
        import re
        score_text = re.sub(r'[^\d.]', '', score_part)
        
        # Convert to float
        if score_text:
            return float(score_text)
        return None
    except Exception as e:
        print(f"Error extracting score: {e}")
        return None


def main():
    client = OpenAI(
        api_key=os.getenv("LKEAP_API_KEY"),  
        base_url="https://api.lkeap.cloud.tencent.com/v1"
    )

    line = [
        "Overall Quality: 2",
        "Intelligibility: 2",
        "Distortion: 2",
        "Distortion Type: artifacts;jitter;timbre & quality",
        "Distortion Duration: appeared between 0 s - 4.9 s (across entire recording)",
        "Distortion Degree: Severe",
        "Distortion Perception: The vocal track is distorted and unclear.",
        "Speech Rate: suitable",
        "Dynamic Range: 2",
        "Emotional Impact: 1",
        "Emotional Impact Type: Neutral",
        "Artistic Expression: 1",
        "Subjective Experience: 3",
        "Subjective Speaker Gender: male",
        "Subjective Speaker Age: middle-aged",
        "Tone Description: The voice sounds middle-aged, hollow and dull.",
        "Improvement Suggestions: emotional impact;expressiveness;timbre;distortion"
    ]

    context = "\n".join(line)
    question = "Please suggest improvements for the speech."
    gt_text = "Reduce distortion artifacts and improve timbre clarity for better intelligibility. Enhance emotional expressiveness to engage listeners more effectively. Refine vocal tone to avoid a hollow, dull sound and improve naturalness."
    pred_text = "Enhance emotional expressiveness to engage listeners more effectively. Reduce distortion artifacts, particularly in the 0s-4.9s range, for clearer vocal quality. Improve artistic expression to make the delivery more vivid and dynamic."
    messages = create_evaluation_prompt(context, question, gt_text, pred_text)

    completion = client.chat.completions.create(
        model="deepseek-v3-0324",
        messages=messages,
        max_tokens=800,
        temperature=0.0,  
        top_p=1.0,         
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False
    )

    evaluation = completion.choices[0].message.content
    score = extract_score(evaluation)
    print("Evaluation Result:")
    print(evaluation)
    print(f"Extracted Score: {score}")

if __name__ == "__main__":
    main()
