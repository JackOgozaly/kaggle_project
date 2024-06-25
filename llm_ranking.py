import pandas as pd
import anthropic
from openai import OpenAI
import os
import numpy as np
from tqdm import tqdm


#Authenticate with Anthropic
api_key = open("/Users/jackogozaly/Desktop/Python_Directory/anthropic_api_key.txt", "r").read().strip("\n")
anthropic_client = anthropic.Client(api_key=api_key)


#Authentic
os.environ["OPENAI_API_KEY"] = open("/Users/jackogozaly/Desktop/Python_Directory/key.txt", "r").read().strip("\n")
#Set up OpenAI Client
openai_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


gpt_4o_df = pd.read_csv(r'/Users/jackogozaly/Downloads/llm-for-environmental-review/prelim_results_openai.csv')

gpt_4_df = pd.read_csv(r'/Users/jackogozaly/Downloads/llm-for-environmental-review/prelim_results_openai_v2.csv')

claude_opus_df = pd.read_csv(r'/Users/jackogozaly/Downloads/llm-for-environmental-review/prelim_results.csv')

sonnet_df = pd.read_csv(r'/Users/jackogozaly/Downloads/llm-for-environmental-review/prelim_results_sonnet.csv')



system_prompt = """
                You are a bot trained to evaluate the quality of answers to a test. You will be given a question,
                some relevant context/ documents to answer that questions, and then 4 different answers.
                
                Your only output should be 1, 2, 3, or 4 
                
                There are no ties, you can only ever pick one best answer. Your goal is to evaluate and find the answer with the
                highest factual correctness. You should also value concision and how human like the response is.
                
                """
                
                
                

def antrhopic_model( 
                    question,
                    context, 
                    answer_1,
                    answer_2,
                    answer_3, 
                    answer_4,
                    system_prompt = system_prompt,
                    model_choice = "claude-3-5-sonnet-20240620"):
    
    
    message = anthropic_client.messages.create(
        model=model_choice,
        max_tokens=1000,
        temperature=1,
        system= system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
                                    Remember: You want to find the answer with the highest accuracy and best writing. Only provide a number associated with the best answer.
                                    
                                    Question: {question}
                                    
                                    Context: {context}
                                    
                                    Answer 1: {answer_1}
                                    
                                    Answer 2: {answer_2}
                                    
                                    Answer 3: {answer_3}
                                    
                                    Answer 4: {answer_4}
                                    
                                    """
                    }
                ]
            }
        ]
        )
    
    return(message.content[0].text)





def openai_model(question,
                    context, 
                    answer_1,
                    answer_2,
                    answer_3, 
                    answer_4,
                    system_prompt = system_prompt,
                    model_choice = "gpt-4o-2024-05-13"):
    
    
    completion = openai_client.chat.completions.create(
        model= model_choice,
        temperature= 1, 
        max_tokens = 1000,
        messages=[
            {"role": "system", "content": f"{system_prompt}"},
            {"role": "user", "content": f"""
                        Remember: You want to find the answer with the highest accuracy and best writing. Only provide a number associated with the best answer.
                        
                        Question: {question}
                        
                        Context: {context}
                        
                        Answer 1: {answer_1}
                        
                        Answer 2: {answer_2}
                        
                        Answer 3: {answer_3}
                        
                        Answer 4: {answer_4}
                        
                        """}
            ]
        )
    
             
    return(completion.choices[0].message.content)


results = []

for i in tqdm(range(len(gpt_4o_df))): 
    
    answer = openai_model(gpt_4o_df['question'].iloc[i], 
                           gpt_4o_df['context'].iloc[i], 
                           gpt_4o_df['PREDICTED_ANSWER'].iloc[i],
                           gpt_4_df['PREDICTED_ANSWER'].iloc[i],
                           claude_opus_df['PREDICTED_ANSWER'].iloc[i],
                           sonnet_df['PREDICTED_ANSWER'].iloc[i]
                           )
    results.append(answer)
              



all_df = pd.DataFrame({'ID': gpt_4o_df['ID'],
                       'question_type': gpt_4o_df['question_type'],
                        'question': gpt_4o_df['question'],
                       'context': gpt_4o_df['context'],
                       'GPT 4o': gpt_4o_df['PREDICTED_ANSWER'],
                       'GPT 4': gpt_4_df['PREDICTED_ANSWER'],
                       'Claude Opus 3': claude_opus_df['PREDICTED_ANSWER'],
                       'Sonnet' : sonnet_df['PREDICTED_ANSWER'],
                       'best_answer': results
    }
    
    )



all_df.to_csv(r'/Users/jackogozaly/Downloads/llm-for-environmental-review/model_ranked_results.csv',
              index=False)


all_df['best_answer'] = all_df['best_answer'].str.replace(r'[^\d.]', '', regex=True)
all_df['best_answer'] = pd.to_numeric(all_df['best_answer'])



final_df = all_df.copy()


final_df['PREDICTED_ANSWER'] = np.where(final_df['best_answer'] == 1,  final_df['GPT 4o'], None)
final_df['PREDICTED_ANSWER'] = np.where(final_df['best_answer'] == 2,  final_df['GPT 4'], final_df['PREDICTED_ANSWER'])
final_df['PREDICTED_ANSWER'] = np.where(final_df['best_answer'] == 3,  final_df['Claude Opus 3'], final_df['PREDICTED_ANSWER'])
final_df['PREDICTED_ANSWER'] = np.where(final_df['best_answer'] == 4,  final_df['Sonnet'], final_df['PREDICTED_ANSWER'])


final_df['PREDICTED_ANSWER'] = np.where(
                                        np.logical_and(final_df['question_type'] == "closed", 
                                                       final_df['PREDICTED_ANSWER'].str.lower().str.contains("yes", regex=False)),
                                                       'Yes', final_df['PREDICTED_ANSWER'])
    
final_df['PREDICTED_ANSWER'] = np.where(
                                        np.logical_and(final_df['question_type'] == "closed", 
                                                       final_df['PREDICTED_ANSWER'].str.lower().str.contains("no", regex=False)),
                                                       'No', final_df['PREDICTED_ANSWER'])

final_df = final_df[['ID', 'PREDICTED_ANSWER']]

final_df.to_csv(r'/Users/jackogozaly/Downloads/llm-for-environmental-review/all_models_ranked.csv', index=False)

