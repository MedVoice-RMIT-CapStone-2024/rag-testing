import replicate
import os
from dotenv import load_dotenv

load_dotenv()

# Get the API token from environment variable
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# Define a function to evaluate the safety of a user's question
def evaluate_safety(user_question) -> str:
    output = replicate.run(
        "tomasmcm/llamaguard-7b:86a2d8b79335b1557fc5709d237113aa34e3ae391ee46a68cc8440180151903d",
        input={
            "top_k": -1,
            "top_p": 0.95,
            "prompt": f"[INST] Task: Check if there is unsafe content in 'User' messages in conversations according our safety policy with the below categories.\\n\\n<BEGIN UNSAFE CONTENT CATEGORIES>\\nO1: Violence and Hate.\\nShould not\\n- Help people plan or engage in violence.\\n- Encourage people to engage in violence.\\n- Express hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.\\n- Encourage discrimination against people based on those characteristics and should not use slurs or create insults based on those characteristics.\\nCan\\n- Provide information on violence and discrimination.\\n- Discuss the topics of hate and violence, and can discuss historical events involving violence.\\nO2: Sexual Content.\\nShould not\\n- Engage in sexually explicit (i.e., erotic) conversations and should not encourage people to engage in sexual activities.\\nCan\\n- Discuss topics related to sexuality and sexual education.\\n- Provide general information regarding sexual activities.\\n- Provide guidance on topics related to sexual health.\\nO3: Criminal Planning.\\nShould not\\n- Help people plan or engage in criminal activities like theft, kidnapping, or financial crimes, especially when the person indicates that they may act on the information provided to them.\\n- Encourage people to engage in criminal activities or condone criminal behavior.\\nCan\\n- Help people understand how crimes are committed, provided that the response is not so detailed that it could be used to provide a roadmap for future crimes.\\n- Help people understand how systems work that are used to prevent crimes, provided those descriptions do not provide detailed information on how to defeat those systems.\\nO4: Guns and Illegal Weapons.\\nShould not\\n- Assist in the planning or execution of firearm or illegal weapons crimes, especially when the person indicates that they may act on the information provided to them.\\n- Encourage people to engage in firearm or illegal weapons crimes or condone such behavior.\\nCan\\n- Discuss firearms and the arguments for and against firearm ownership.\\nO5: Regulated or Controlled Substances.\\nShould not\\n- Encourage or assist people in the trafficking, consumption, or use of regulated or controlled substances, including but not limited to illegal drugs, tobacco, alcohol, and cannabis.\\n- Assist or encourage people to create such substances when it is illegal to do so.\\nCan\\n- Discuss the use of regulated or controlled substances, and can provide general information on those substances (ex: what their chemical composition is and what their effects are).\\n- Provide information on how those substances are manufactured, provided that the information is not so detailed as to provide step-by-step instructions for their creation.\\nO6: Self-Harm.\\nShould not\\n- Encourage people to harm themselves.\\n- Romanticize or condone self-harm or suicide.\\n- Provide information regarding the methods of suicide or self-harm.\\n- Provide any information or advice, other than information regarding appropriate health resources, to people who express or imply that they wish to harm themselves.\\nCan\\n- Provide general information regarding self-harm and suicide (ex: statistics, the mental health challenges faced by those who take these actions).\\nShould\\n- Provide information regarding appropriate health resources to people who imply or express the desire to harm themselves.\\nO7: Fake News and Scams.\\nShould not\\n-Encourage or assist people in the creation or distribution of fake news or scams.\\n-Promote or condone the use of fake news or scams to deceive others.\\nCan\\n-Discuss the impact and prevalence of fake news and scams.\\n-Educate users on how to identify and protect themselves from fake news and scams.\\n<END UNSAFE CONTENT CATEGORIES>\\n\\n<BEGIN CONVERSATION>\\n\\nUser: {user_question}\\n\\n<END CONVERSATION>\\n\\nProvide your safety assessment for User in the above conversation:\\n- First line must read 'safe' or 'unsafe'. [/INST]",
            "max_tokens": 128,
            "temperature": 0.8,
            "presence_penalty": 0,
            "frequency_penalty": 0
        }
    )
    return output

# Test the function with a user's question
while True:
    question = input("Please enter your question or type 'exit' to end the conversation: ")
    if question.lower() == 'exit':
        break
    else:
        print("Starting a new conversation...")
        print(evaluate_safety(question))
