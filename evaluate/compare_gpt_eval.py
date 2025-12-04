import os
from openai import OpenAI


def create_evaluation_prompt(context, question, answer_1, answer_2):
    return [
        {
            'role': 'user',
            'content': f'''You are an evaluator assessing the quality of an AI-generated response based on a provided human-labeled context and a human-written ground-truth response.

### Context:
{context}

### Question:
{question}

### Answer_1 (Ground Truth Answer):
{answer_1}

### Answer_2 (Model-generated Answer):
{answer_2}

### Task:
1. Extract the concluding judgments from both Answer_1 and Answer_2 for the following aspects:
   - **Overall Quality**: Which answer (A or B) is better, or are they the same?
   - **Intelligibility**: Which answer has better intelligibility, or are they the same?
   - **Distortion**: Which answer has less distortion, or are they the same?
   - **Speech Rate**: Which answer has a better speech rate, or are they the same?
   - **Dynamic Range**: Which answer shows better dynamic range, or are they the same?
   - **Emotional Impact**: Which answer has better emotional impact, or are they the same?
   - **Artistic Expression**: Which answer has better artistic expression, or are they the same?
   - **Subjective Experience**: Which answer provides a better subjective experience, or are they the same?

2. For each aspect, compare the extracted conclusions:
   - If the conclusions for an aspect are the same (e.g., both say Answer_1 is better), mark them as "consistent."
   - If the conclusions for an aspect differ (e.g., one says Answer_1 is better and the other says Answer_2 is better), or if sentence 2 contains no relevant content, mark them as "inconsistent."

3. Evaluate the overall quality of Answer_2 based on the following general aspects:
   - Helpfulness: Does the answer provide useful and relevant information?
   - Relevance: Does it stay on topic and align with the context?
   - Accuracy: Does it reflect the facts stated in the context?
   - Level of Detail: Is the response thorough and precise?

4. Provide a brief explanation of your reasoning, comparing Answer_2 with Answer_1. After that:
   - Provide a simple numerical score (0 to 10) for Answer_2 based on its quality relative to Answer_1.
   - For each aspect (Overall Quality, Intelligibility, Distortion, Speech Rate, Dynamic Range, Emotional Impact, Artistic Expression, and Subjective Experience), indicate whether the conclusions are "consistent" or "inconsistent."
   - Display the extracted conclusion for each aspect from both Answer_1 and Answer_2.

### Important Output Formatting Guidelines:
- All eight dimensions must be extracted. 
- Use the **exact phrasing** for each field. For example, **Conclusion Consistency for Overall Quality: Consistent** must appear exactly as shown.
- There should be no extra words, punctuation, or deviations from this format.

### Output Format:
# Explanation: <Your simple reasoning here>  
Score: <A number from 0 to 10>  
Conclusion Consistency for Overall Quality: <Consistent/Inconsistent>  
Extracted Conclusion from Answer_1 (Overall Quality): <The extracted conclusion sentence from Answer_1 for Overall Quality>  
Extracted Conclusion from Answer_2 (Overall Quality): <The extracted conclusion sentence from Answer_2 for Overall Quality>  

Conclusion Consistency for Intelligibility: <Consistent/Inconsistent>  
Extracted Conclusion from Answer_1 (Intelligibility): <The extracted conclusion sentence from Answer_1 for Intelligibility>  
Extracted Conclusion from Answer_2 (Intelligibility): <The extracted conclusion sentence from Answer_2 for Intelligibility>  

Conclusion Consistency for Distortion: <Consistent/Inconsistent>  
Extracted Conclusion from Answer_1 (Distortion): <The extracted conclusion sentence from Answer_1 for Distortion>  
Extracted Conclusion from Answer_2 (Distortion): <The extracted conclusion sentence from Answer_2 for Distortion>

Conclusion Consistency for Speech Rate: <Consistent/Inconsistent>  
Extracted Conclusion from Answer_1 (Speech Rate): <The extracted conclusion sentence from Answer_1 for Speech Rate>  
Extracted Conclusion from Answer_2 (Speech Rate): <The extracted conclusion sentence from Answer_2 for Speech Rate>  

Conclusion Consistency for Dynamic Range: <Consistent/Inconsistent>  
Extracted Conclusion from Answer_1 (Dynamic Range): <The extracted conclusion sentence from Answer_1 for Dynamic Range>  
Extracted Conclusion from Answer_2 (Dynamic Range): <The extracted conclusion sentence from Answer_2 for Dynamic Range>  

Conclusion Consistency for Emotional Impact: <Consistent/Inconsistent>  
Extracted Conclusion from Answer_1 (Emotional Impact): <The extracted conclusion sentence from Answer_1 for Emotional Impact>  
Extracted Conclusion from Answer_2 (Emotional Impact): <The extracted conclusion sentence from Answer_2 for Emotional Impact>  

Conclusion Consistency for Artistic Expression: <Consistent/Inconsistent>  
Extracted Conclusion from Answer_1 (Artistic Expression): <The extracted conclusion sentence from Answer_1 for Artistic Expression>  
Extracted Conclusion from Answer_2 (Artistic Expression): <The extracted conclusion sentence from Answer_2 for Artistic Expression>  

Conclusion Consistency for Subjective Experience: <Consistent/Inconsistent>  
Extracted Conclusion from Answer_1 (Subjective Experience): <The extracted conclusion sentence from Answer_1 for Subjective Experience>  
Extracted Conclusion from Answer_2 (Subjective Experience): <The extracted conclusion sentence from Answer_2 for Subjective Experience>'''
        }
    ]


def extract_score(evaluation_text):
    """Extract numerical score from GPT evaluation text."""
    if not evaluation_text:
        return None
    
    try:
        evaluation_text = evaluation_text.replace('\n', ' ')
        if "**Score:**" in evaluation_text:
            score_pattern = "**Score:**"
        elif "Score:" in evaluation_text:
            score_pattern = "Score:"
        else:
            return None
        score_part = evaluation_text.split(score_pattern)[1].strip()
        import re
        if '/' in score_part:
            match = re.search(r'(\d+)/10', score_part)
            if match:
                return float(match.group(1))
        match = re.search(r'(\d+(?:\.\d+)?)', score_part)
        if match:
            return float(match.group(1))
        score_text = re.sub(r'[^\d.]', '', score_part)
        if score_text:
            return float(score_text)
        return None
    except Exception as e:
        print(f"Error extracting score: {e}")
        return None

def extract_multi_dimensional_consistency(evaluation_text):
    """Extract consistency for multiple dimensions from GPT evaluation text."""
    if not evaluation_text:
        return {}
    try:
        dimensions = [
            "Overall Quality",
            "Intelligibility", 
            "Distortion",
            "Speech Rate",
            "Dynamic Range",
            "Emotional Impact",
            "Artistic Expression",
            "Subjective Experience"
        ]
        consistency_results = {}
        import re
        for dimension in dimensions:
            pattern1 = rf"Conclusion Consistency for {re.escape(dimension)}:\s*(?:\*\*)?(Consistent|Inconsistent|Fail)(?:\*\*)?"
            match1 = re.search(pattern1, evaluation_text, re.IGNORECASE)
            if match1:
                consistency_results[dimension] = match1.group(1)
                continue
            pattern2 = rf"#### {re.escape(dimension)}:\s*[\r\n]+(?:\*\*Conclusion Consistency\*\*:\s*(Consistent|Inconsistent|Fail))"
            match2 = re.search(pattern2, evaluation_text, re.IGNORECASE)
            if match2:
                consistency_results[dimension] = match2.group(1)
                continue
            pattern3 = rf"\*\*Conclusion Consistency for {re.escape(dimension)}\*\*:\s*(Consistent|Inconsistent|Fail)"
            match3 = re.search(pattern3, evaluation_text, re.IGNORECASE)
            if match3:
                consistency_results[dimension] = match3.group(1)
                continue
            pattern4 = rf"#### {re.escape(dimension)}:.*?\*\*Conclusion Consistency\*\*:\s*(Consistent|Inconsistent|Fail)"
            match4 = re.search(pattern4, evaluation_text, re.IGNORECASE | re.DOTALL)
            if match4:
                consistency_results[dimension] = match4.group(1)
                continue
            pattern5 = rf"####\s*\*\*{re.escape(dimension)}\*\*.*?- \*\*Conclusion Consistency\*\*:\s*(Consistent|Inconsistent|Fail)"
            match5 = re.search(pattern5, evaluation_text, re.IGNORECASE | re.DOTALL)
            if match5:
                consistency_results[dimension] = match5.group(1)
                continue
            pattern6 = rf"\*\*Conclusion Consistency for {re.escape(dimension)}:\*\*\s*\n\s*\*\*(Consistent|Inconsistent|Fail)\*\*"
            match6 = re.search(pattern6, evaluation_text, re.IGNORECASE)
            if match6:
                consistency_results[dimension] = match6.group(1)
                continue
            pattern7 = rf"\*\*Conclusion Consistency for {re.escape(dimension)}:\*\*\s*(Consistent|Inconsistent|Fail)"
            match7 = re.search(pattern7, evaluation_text, re.IGNORECASE)
            if match7:
                consistency_results[dimension] = match7.group(1)
                continue
            consistency_results[dimension] = None
        return consistency_results
    except Exception as e:
        print(f"Error extracting multi-dimensional consistency: {e}")
        return {}

def main():
    client = OpenAI(
        api_key=os.getenv("LKEAP_API_KEY"),
        base_url="https://api.lkeap.cloud.tencent.com/v1"
    )

    lines = [
    "Overall Quality (A vs B): A and B are similar in this aspect",
    "Intelligibility (A vs B): A and B are similar in this aspect",
    "Distortion (A vs B): A and B are similar in this aspect",
    "Speech Rate (A vs B): A and B are similar in this aspect",
    "Dynamic Range (A vs B): A and B are similar in this aspect",
    "Emotional Impact (A vs B): B is better than A",
    "Artistic Expression (A vs B): B is better than A",
    "Subjective Experience (A vs B): B is better than A",
    "Distortion Detail A - Type: jitter, Duration: 0-3.3 s (throughout the recording)",
    "Distortion Detail B - Type: jitter, Duration: 0-3.3 s (throughout the recording)",
    "Emotional Impact Type A: Neutral",
    "Emotional Impact Type B: Neutral",
    "Subjective A - Gender: female, Age: middle-aged",
    "Subjective B - Gender: female, Age: middle-aged"
    ]
    context = "\n".join(lines)
    question = "Describe the speech in detail from the quality aspects."
    gt_text = "Overall, Samples A and B demonstrate similar general quality, intelligibility, speech rate, and dynamic range. However, both samples exhibit severe jitter distortion from 0 to 3.3 seconds, causing a strong, shaky perception throughout the recording. While their objective technical aspects are comparable, Sample B outperforms A in subjective expressiveness, delivering better emotional impact (though both are neutral), artistic expression, and overall listening experience. Both speakers are female, in their thirties, with rough and dull tones, but Sample B’s superior expressiveness makes it more engaging despite similar technical flaws."
    pred_text = "Overall, Samples A and B exhibit similar quality in terms of clarity, speech rate, and dynamic range. Both recordings suffer from severe jitter distortion throughout their durations (A: 0–3.3 s; B: 0–3.2 s), causing a strong, shaky perception that affects naturalness. Despite this, intelligibility remains comparable. Subjectively, neither sample conveys distinct emotions (both neutral), and artistic expression is equally matched. The speakers—both female, in their thirties, with rough and dull tones—contribute to a similar listening experience. In summary, while technical flaws persist, the overall quality and expressiveness of both samples are closely aligned."
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
    multi_consistency = extract_multi_dimensional_consistency(evaluation)
    print("Evaluation Result:")
    print(evaluation)
    print(f"Extracted Score: {score}")
    print(f"Extracted Multi-dimensional Consistency: {multi_consistency}")
    

if __name__ == "__main__":
    main()
