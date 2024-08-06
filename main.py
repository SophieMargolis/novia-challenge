
import json
# pip install PyPDF2
import PyPDF2
# pip install requests
import requests


def read_config():
    with open('config.json') as f:
        data = json.load(f)
    return data['openai']['api_key']

# PART 1

def extract_text_from_pdf(pdf_path):
    text = ''
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        for page_num in range(reader.numPages):
            page = reader.getPage(page_num)
            text += page.extract_text()
    return text


def construct_cv_prompt(cv_text):
    prompt = (
        "Extract the following information from the candidate's CV:\n"
        "1. Candidate's name\n"
        "2. Key skills (list of 5-10 most relevant skills)\n"
        "3. Years of experience\n"
        "4. Education level\n"
        "5. Most recent job title and company\n"
        "6. A brief summary of the candidate's profile (2-3 sentences).\n\n"
        "CV Text:\n"
        f"{cv_text}\n"
        "Provide the information in the following JSON format:\n"
        "{\n"
        "  'name': 'string',\n"
        "  'skills': ['string', 'string', ...],\n"
        "  'years_of_experience': 'number',\n"
        "  'education_level': 'string',\n"
        "  'most_recent_job': 'string',\n"
        "  'company': 'string',\n"
        "  'profile_summary': 'string'\n"
        "}\n"
    )
    return prompt

def call_llm_api(prompt, api_key):
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    data = {
        'model': 'gpt-4',  # or 'gpt-3.5-turbo'
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.5 # We can adjust the creativity of the model
    }
    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    return result['choices'][0]['message']['content']

def cv_analysis(pdf_path, api_key):
    # Extract text from the CV PDF
    cv_text = extract_text_from_pdf(pdf_path)
    
    # Construct the prompt for the LLM API
    prompt = construct_cv_prompt(cv_text)
    
    # Call the LLM API and get the response
    api_response = call_llm_api(prompt, api_key)
    
    # Print or return the structured information
    print("Extracted Information:\n", api_response)
    return api_response


# PART 2

def construct_fit_prompt(job_description, candidate_info):
    prompt = (
        "Evaluate the candidate's fit for the following job description based on their CV information:\n"
        "Job Description:\n"
        f"{job_description}\n\n"
        "Candidate Information:\n"
        f"{candidate_info}\n\n"
        "Provide a summary about the candidate's match to the role and classify their fit into one of the following categories:\n"
        "A: Good fit\n"
        "B: Medium fit\n"
        "C: Not a good fit\n"
        "Provide the information in the following JSON format:\n"
        "{\n"
        "  'fit_summary': 'string',\n"
        "  'fit_category': 'string'\n"
        "}\n"
    )
    return prompt

def evaluate_candidate_fit(job_description, candidate_info, api_key):
    # Construct the prompt for the LLM API
    prompt = construct_fit_prompt(job_description, candidate_info)
    
    # Call the LLM API and get the response
    api_response = call_llm_api(prompt, api_key)
    
    # Print or return the fit evaluation
    print("Fit Evaluation:\n", api_response)
    return api_response



# Usage of all functions
if __name__ == "__main__":
    config = read_config()
    api_key = config['api_key']
    
    # part 1 function
    pdf_path = 'path/to/your/candidate_cv.pdf'
    cv_info = cv_analysis(pdf_path, api_key)
    
    # part 2 function
    job_description = "Job description text here"
    fit_evaluation = evaluate_candidate_fit(job_description, cv_info, api_key)

