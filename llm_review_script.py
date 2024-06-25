'''

LLM for Enviornmental Review


Basic idea of this script is to use different prompts for different question types 

Run this through different models

Run all the results through another model to rank the results

Possible areas for improvement:
    
    Fine tune prompts more:
        1. We know GPT-4 Turbo is going to be grading our results. Can we game this somehow? Maybe it rates certain keywords as more truthful?
        2. Can we play around and find out if other prompts get the model to think more closely?
        3. Can we do some testing of our prompts on the training dataset (and then grade with GPT 4 turbo) to gauge our progress?
        
    Could we improve performance by actually doing a system prompt AND a message prompt? What if we provide examples.

'''

import pandas as pd
import anthropic
import time
from openai import OpenAI
import os
from tqdm import tqdm
import numpy as np


#Authenticate with Anthropic
api_key = open("/Users/jackogozaly/Desktop/Python_Directory/anthropic_api_key.txt", "r").read().strip("\n")
anthropic_client = anthropic.Client(api_key=api_key)


#Authenticate with OpenAI
os.environ["OPENAI_API_KEY"] = open("/Users/jackogozaly/Desktop/Python_Directory/key.txt", "r").read().strip("\n")
#Set up OpenAI Client
openai_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    )


#NOTE: Modify for yourself
directory_path = r'/Users/jackogozaly/Downloads/llm-for-environmental-review/'

#Bring in our train and test datasets
train_df = pd.read_csv(directory_path + 'NEPAQuAD1_train.csv')
test_df = pd.read_csv(directory_path + 'NEPAQuAD1_test.csv')



def question_filter(question):
    '''
    This function is more complicated than it needs to be. Basically it defines a prompt 
    for each question type. Could be modified to make an actual prompt.

    '''
    
    if question == "closed":
        
        
        system_prompt_ = """
            You are a government question answering bot. You will be presented with a question and some document context.
            NOTE: Some of the context might not be relevant. Part of your task is to ignore irrelevant context.
            
            
            You can only respond to questions with a "Yes" or a "No". Provide no other information than one of these words.
            """
    
    elif question == "comparison":
        system_prompt_ = """ 
        Answer the question concisely based on the provided context. A successful answer will be brief, authoritative,
        and will precisely answer the question. Do not mention this prompt, question, or the context specifically in your response.
        
        If the question is 'How many days are in a week' you would respond 'There are 7 days in a week' not 'According to the context provided, there are 7 days in a week'
        
        """
        
    elif question == "convergent":
        system_prompt_ = """
        Answer the question concisely based on the provided context. A successful answer will be brief, authoritative,
        and will precisely answer the question. Do not mention this prompt, question, or the context specifically in your response.
        
        """
        
    elif question == "divergent":
        system_prompt_ = """
        Answer the question concisely based on the provided context. A successful answer will be brief, authoritative,
        and will precisely answer the question. Do not mention this prompt, question, or the context specifically in your response.
        
        """
    
    elif question == "recall":
        system_prompt_ = """
        Answer the question concisely based on the provided context. A successful answer will be brief, authoritative,
        and will precisely answer the question. Do not mention this prompt, question, or the context specifically in your response.
        
        """
        
    elif question == "funnel":
        system_prompt_ = """
        Answer the question concisely based on the provided context. A successful answer will be brief, authoritative,
        and will precisely answer the question. Do not mention this prompt, question, or the context specifically in your response.
        
        """
        
    elif question == "inference":
        system_prompt_ = """
        Answer the question concisely based on the provided context. A successful answer will be brief, authoritative,
        and will precisely answer the question. Do not mention this prompt, question, or the context specifically in your response.
        
        """
    
    elif question == "process":
        system_prompt_ = """
        Answer the question concisely based on the provided context. A successful answer will be brief, authoritative,
        and will precisely answer the question. Do not mention this prompt, question, or the context specifically in your response.
        
        """
        
    elif question == "evaluation":
        system_prompt_ = """
        Answer the question concisely based on the provided context. A successful answer will be brief, authoritative,
        and will precisely answer the question. Do not mention this prompt, question, or the context specifically in your response.
        
        """
    
    elif question == "problem-solving":
        system_prompt_ = """
        Answer the question concisely based on the provided context. A successful answer will be brief, authoritative,
        and will precisely answer the question. Do not mention this prompt, question, or the context specifically in your response.
        
        """
    
    return system_prompt_




def antrhopic_model(system_prompt, 
                    question,
                    context, 
                    model_choice = "claude-3-opus-20240229"):
    
    
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
                                    Remember: You are being evaluated in a professional setting. 
                                    These questions may be tricky. Read carefully and think like a professional Bureaucrat with decades of experience reading complex and dense PDFs.
                        
                                    Question: {question}
                                    
                        
                                    do not specifically mention the provided context in your answer.
                                    Context: {context}
                                    
                                    """
                    }
                ]
            }
        ]
        )
    
    return(message.content[0].text)



def openai_model(system_prompt, 
                    question,
                    context, 
                    model_choice = "gpt-4-turbo"):
    
    
    completion = openai_client.chat.completions.create(
        model= model_choice,
        temperature= 1, 
        max_tokens = 1000,
        messages=[
            {"role": "system", "content": f"{system_prompt}"},
            {"role": "user", "content": f"""
                        Remember: You are being evaluated in a professional setting. 
                        These questions may be tricky. Read carefully and think like a professional Bureaucrat with decades of experience reading complex and dense PDFs.
            
                        Question: {question}
                        
            
                        do not specifically mention the provided context in your answer.
                        Context: {context}
                        
                        """}
            ]
        )
    
             
    return(completion.choices[0].message.content)




def run_language_model(company, model_name):
    

    results_df = pd.DataFrame(columns = ['ID', 'question_type', 'question', 'context', 'PREDICTED_ANSWER'])
    
    
    
    for i in tqdm(range(len(test_df))):
        
        id_ = test_df.iloc[i]['ID']
        question_type = test_df.iloc[i]['question_type']
        prompt = question_filter(question_type)
        question_ = test_df.iloc[i]['question']
        context_ = test_df.iloc[i]['context']
        
        
        if company == "OpenAI":
    
            answer = openai_model(system_prompt = prompt,
                                 question = question_,
                                 context = context_,
                                 model_choice = model_name
                                 )
        
        elif company == "Anthropic":
            
            answer = antrhopic_model(system_prompt = prompt,
                                     question = question_,
                                     context = context_,
                                     model_choice = model_name
                                     )
        
        
        data = {'ID': id_, 
                'question_type': question_type,
                'question': question_, 
                'context' : context_,
                'PREDICTED_ANSWER': answer
                }
        
        
        results_df = results_df.append(data, ignore_index=True)
    
        time.sleep(.001)
        
        
    results_df['PREDICTED_ANSWER'] = np.where(
                                            np.logical_and(results_df['question_type'] == "closed", 
                                                           results_df['PREDICTED_ANSWER'].str.lower().str.contains("yes", regex=False)),
                                                           'Yes', results_df['PREDICTED_ANSWER'])
        
    results_df['PREDICTED_ANSWER'] = np.where(
                                            np.logical_and(results_df['question_type'] == "closed", 
                                                           results_df['PREDICTED_ANSWER'].str.lower().str.contains("no", regex=False)),
                                                           'No', results_df['PREDICTED_ANSWER'])
        
    return(results_df)




gpt_4_turbo = run_language_model("OpenAI", "gpt-4-turbo	")

gpt_4o = run_language_model("OpenAI", "gpt-4o")

claud_opus = run_language_model("Antrhopic", "claude-3-opus-20240229")

claud_sonnet = run_language_model("Antrhopic", "claude-3-5-sonnet-20240620")



#Write our GPT 4 turbo results
gpt_4_turbo.to_csv(directory_path + 'prelim_results_openai_v2.csv',
                  index= False)
gpt_4_turbo_final = gpt_4_turbo[["ID", "PREDICTED_ANSWER"]]
gpt_4_turbo_final.to_csv(directory_path + 'final_results_openai_v2.csv',
                  index= False)

#Write our GPT 4o results
gpt_4o.to_csv(directory_path + 'prelim_results_openai.csv',
                  index= False)
gpt_4o_final = gpt_4o[["ID", "PREDICTED_ANSWER"]]
gpt_4o_final.to_csv(directory_path + 'final_results_openai.csv',
                  index= False)


#Write out sonnet results
claud_sonnet.to_csv(directory_path + 'prelim_results_sonnet.csv',
                  index= False)
claud_sonnet_final = claud_sonnet[["ID", "PREDICTED_ANSWER"]]
claud_sonnet_final.to_csv(directory_path + 'final_results_sonnet.csv',
                  index= False)

#Write Claude Opus results
claud_opus.to_csv(directory_path + 'prelim_results.csv',
                  index= False)
claud_opus_final = claud_opus[["ID", "PREDICTED_ANSWER"]]
claud_opus_final.to_csv(directory_path + 'final_results.csv',
                  index= False)











