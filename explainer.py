from openai import OpenAI
client = OpenAI()

 """
    Transforms prediction results into a plain-language explanation using GPT-4o.
    High-confidence results are listed first, followed by low-confidence ones.
    
"""

def explain_prediction(results):
    top_anomalies = [(label, prob) for label, prob in results if prob >= 0.5]
    low_anomalies = [(label, prob) for label, prob in results if 0 < prob < 0.5]

    top_str = ', '.join([f"{label} ({prob*100:.2f}%)" for label, prob in top_anomalies])
    low_str = ', '.join([f"{label} ({prob*100:.2f}%)" for label, prob in low_anomalies])

    prompt = (
        f"We analyzed a retinal image and detected the following conditions with high confidence: {top_str}.\n"
        f"There are also signs of these conditions with lower probability: {low_str}.\n"
        f"Now, state what we suspect with the confidence rates then explain that its likely something .\n"
        f"Explain in 3 simple sentences what these anomalies mean and why they are important for vision and health."
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=250
    )
    return response.choices[0].message.content.strip()

